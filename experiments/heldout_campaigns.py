"""Evaluate frozen models on the remaining annotated campaign windows."""
import argparse
import csv
import json
from pathlib import Path
import time
from types import SimpleNamespace
import sys

import numpy as np

from add_stress_views import stream_cache, utc_ns
from causal_data import EventDay, positive_processes
from evaluation import process_metrics


ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/full_grid_v1'
WINDOWS=(('theia','2018-04-12','16:00','18:00'),('cadets','2018-04-12','17:30','19:30'),
         ('cadets','2018-04-13','12:30','14:30'))


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def put(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def setup():
    for dataset,date,begin,end in WINDOWS:
        source=ROOT/('data/processed/'+dataset+'_e3')
        original=ROOT/('data/cache/'+dataset+'_longoutage_v1')
        split=load(original/'split.json')
        day=EventDay(source,date,split['context'],split['lookback_seconds'])
        start_iso,end_iso=[date+'T'+value+':00Z' for value in (begin,end)]
        begin_ns,end_ns=map(utc_ns,[start_iso,end_iso])
        targets=day.target_range(begin_ns,end_ns)
        if not len(targets):raise RuntimeError('Missing held-out campaign '+date)
        root=ROOT/'data/cache/heldout'/dataset/date
        stream_cache(day,targets,np.load(original/'token_map.npy'),root,
                     dict(mode='clean',interval_ns=[end_ns,end_ns],relation_id=None))
        actors=day.events['actor'][targets];objects=day.events['object'][targets]
        universe=np.unique(np.r_[actors,objects[day.events['otype'][targets]==1]])
        positive=positive_processes(source,date)
        cohort=dict(dataset=dataset,date=date,window=[start_iso,end_iso],universe=universe.tolist(),
                    positive_processes=sorted(positive&set(universe)),outside_window_positives=sorted(positive-set(universe)),
                    events=len(targets),actor_count=len(np.unique(actors)))
        put(root/'cohort.json',cohort)
        day.close()
        yield dataset,date,root,source,cohort


def stats_scores(cache,probabilities,method):
    scores={}
    contexts,lengths,targets,actors=(np.load(cache/(name+'.npy'),mmap_mode='r') for name in ('contexts','lengths','targets','actors'))
    marginal,transition=probabilities['marginal'],probabilities['transition']
    for offset in range(0,len(targets),131072):
        end=min(len(targets),offset+131072)
        if method=='frequency':values=-np.log(marginal[targets[offset:end]])
        else:
            prev=contexts[np.arange(offset,end),lengths[offset:end].astype(np.int64)-1]
            values=-np.log(transition[prev,targets[offset:end]])
        actor_ids=actors[offset:end]
        cuts=np.r_[0,np.flatnonzero(actor_ids[1:]!=actor_ids[:-1])+1]
        for actor,value in zip(actor_ids[cuts],np.maximum.reduceat(values,cuts)):
            scores[int(actor)]=max(scores.get(int(actor),0.),float(value))
    return scores


def graph_scores(method,cache,source,date,primary,device):
    import torch
    from prepare_dataset import REVERSED
    schema=load(source/'schema.json')
    reverse=[int(k) for k,v in schema['relation_names'].items() if v in REVERSED]
    kinds=np.load(source/'node_types.npy',mmap_mode='r')
    indices=np.load(cache/'indices.npy',mmap_mode='r')
    actors=np.unique(np.load(cache/'actors.npy',mmap_mode='r'))
    day=EventDay(source,date)
    config=load(primary/method/'results.json')
    model_path=primary/method
    try:
        if method=='magic_adapted':
            from magic_adapted import GraphModel,graphs,score
            net=GraphModel().to(device)
            net.load_state_dict(torch.load(model_path/'checkpoint.pt',map_location=device)['model'])
            net.eval()
            reference=torch.from_numpy(np.load(model_path/'reference.npy')).to(device)
            values,_=score(net,graphs(day,indices,reverse,config['config']['cap'],kinds),device,reference,actors)
            return values,config['threshold']
        from protrap_adapted import UPSTREAM,embeddings,shards
        sys.path.insert(0,str(UPSTREAM))
        from model.autoencoder_cl import build_model
        mapping=np.load(ROOT/('data/cache/'+config_for_dataset(source)+'_longoutage_v1/token_map.npy'))
        known={int(raw)//4 for raw in np.flatnonzero(mapping!=2)}
        rels={value:index+1 for index,value in enumerate(sorted(known))}
        lookup=np.array([rels.get(index,0) for index in range(len(schema['relation_names']))])
        dimension=len(known)+1
        net=build_model(SimpleNamespace(num_hidden=64,num_layers=3,negative_slope=.2,mask_rate=.5,
                                        alpha_l=3,n_dim=3,e_dim=dimension,sr=0))
        net.load_state_dict(torch.load(model_path/'checkpoint.pt',map_location='cpu'))
        net.eval()
        reference=torch.from_numpy(np.load(model_path/'reference.npy'))
        graph_data=((nodes,types,src,dst,lookup[edge_type],times)
                    for nodes,types,src,dst,edge_type,times in shards(day,indices,reverse,kinds,config['cap']))
        scores={}
        for nodes,vectors in embeddings(net,graph_data,dimension,actors):
            for offset in range(0,len(nodes),256):
                distances=torch.cdist(vectors[offset:offset+256],reference)
                values=distances.topk(min(5,len(reference)),largest=False).values.mean(1).numpy()
                for node,value in zip(nodes[offset:offset+256],values):
                    scores[int(node)]=max(scores.get(int(node),0.),float(value))
        return scores,config['threshold']
    finally:
        day.close()


def config_for_dataset(source):
    return source.name.rsplit('_',1)[0]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    cohorts=list(setup())
    if args.prepare_only:return
    import torch
    import joblib
    from count_baseline import score as count_score
    from train_eval import EventGRU,cache_arrays,score_cache
    torch.set_num_threads(2)
    device=torch.device('cuda')
    rows=[]
    for dataset,date,cache,source,cohort in cohorts:
        output=RUN/'heldout'/dataset/date
        output.mkdir(parents=True,exist_ok=True)
        primary=RUN/dataset
        def record(method,seed,scores,threshold,seconds):
            result=process_metrics(cohort['universe'],cohort['positive_processes'],scores,threshold=threshold)
            put(output/(str(seed)+'_'+method+'.json'),dict(metrics=result,scores=scores,seconds=seconds))
            rows.append(dict(dataset=dataset,date=date,method=method,seed=seed,events=cohort['events'],
                processes=result['processes'],positives=result['positive_processes'],
                scoreable_positives=result['positive_scoreable'],AP_all=result['AP_all'],
                recall_at_50=result['top_50']['recall'],precision_at_50=result['top_50']['precision'],
                tp_threshold=result['frozen_threshold']['tp'],fp_threshold=result['frozen_threshold']['fp'],seconds=seconds))
        baseline=load(primary/'statistics/results.json')['results']
        with np.load(primary/'statistics/fitted_probabilities.npz') as probs:
            for method in ('frequency','markov'):
                start=time.monotonic()
                scores=stats_scores(cache,probs,method)
                record(method,17,scores,baseline[method]['threshold'],time.monotonic()-start)
        count_log=load(primary/'isolation_forest/results.json')
        start=time.monotonic()
        count_model=joblib.load(primary/'isolation_forest/model.joblib')
        record('isolation_forest',17,count_score(cache,count_model,count_log['vocabulary']),
               count_log['threshold'],time.monotonic()-start)
        arrays=cache_arrays(cache)
        for seed in (17,23,42):
            for mode in ('clean','iid','dispersed','crop','block'):
                folder=primary/f'{seed}_{mode}'
                original=load(folder/'results.json')
                checkpoint=torch.load(folder/'best.pt',map_location=device)
                conf=checkpoint['config']
                model=EventGRU(conf['dataset']['vocabulary_size'],conf['embedding'],conf['hidden']).to(device)
                model.load_state_dict(checkpoint['model'])
                scores,resource=score_cache(model,arrays,device,1024,output/f'{seed}_{mode}_event_nll.npy')
                record('GRU_'+mode,seed,scores,original['validation']['threshold'],resource['seconds'])
        for method in ('magic_adapted','protrap_adapted'):
            begin=time.monotonic()
            scores,threshold=graph_scores(method,cache,source,date,primary,device)
            record(method,17,scores,threshold,time.monotonic()-begin)
        put(output/'cohort.json',cohort)
        print(json.dumps(dict(dataset=dataset,date=date,events=cohort['events'],positives=len(cohort['positive_processes']))),flush=True)
    with (RUN/'heldout_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='COMPLETED',rows=len(rows))),flush=True)


if __name__=='__main__':main()
