"""Resource-bounded MAGIC adaptation using PIDSMaker encoder and objectives."""
import argparse
import importlib.util
import inspect
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch import nn

from causal_data import EventDay, NS
from evaluation import process_metrics
from prepare_dataset import REVERSED
from train_eval import seed_everything


ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'external/PIDSMaker'


def upstream_module(name,relative):
    spec=importlib.util.spec_from_file_location(name,UPSTREAM/'pidsmaker'/relative)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GraphModel(nn.Module):
    def __init__(self):
        super().__init__()
        module=upstream_module('pid_magic','encoders/magic_encoder.py')
        legacy=module.GATConv
        if 'edge_dim' not in inspect.signature(legacy.__init__).parameters:
            class TypeOnlyGAT(legacy):
                def __init__(self,*args,edge_dim=None,**kwargs):
                    if edge_dim is not None:raise ValueError('This adapter is type only')
                    super().__init__(*args,**kwargs)
                def forward(self,x,edge_index,edge_attr=None):
                    if edge_attr is not None:raise ValueError('Unexpected edge features')
                    return super().forward(x,edge_index)
            module.GATConv=TypeOnlyGAT
        gat=module.MagicGAT
        losses=upstream_module('pid_losses','losses.py')
        feature=upstream_module('pid_feat','objectives/reconstruct_masked_feat.py').GMAEFeatReconstruction
        structure=upstream_module('pid_struct','objectives/predict_masked_struct.py').GMAEStructPrediction
        self.encoder=gat(3,64,64,3,4,feat_drop=0.,activation=nn.PReLU(),concat_out=True)
        decoder=gat(64,64,3,1,1,feat_drop=0.,is_decoder=True)
        self.feature=feature(decoder,losses.sce_loss,.5,3)
        self.structure=structure(nn.Sequential(nn.Linear(128,4),nn.LeakyReLU(),nn.Linear(4,1)),
                                 losses.binary_cross_entropy)

    def loss(self,features,edges):
        masked,ids=self.feature.mask_input(features)
        hidden=self.encoder(masked,edges)['h']
        return (self.feature(features,hidden,edges,False,mask_nodes=ids)['loss']+
                self.structure(hidden,edges,False)['loss'])


def graphs(day, indices, reversed_ids, cap, kinds):
    times=day.events['t'][indices]
    order=np.argsort(times,kind='stable')
    ordered=indices[order]
    buckets=times[order]//(900*NS)
    boundaries=np.r_[0,np.flatnonzero(buckets[1:]!=buckets[:-1])+1,len(ordered)]
    for first,last in zip(boundaries[:-1],boundaries[1:]):
        for offset in range(first,last,cap):
            selected=ordered[offset:min(last,offset+cap)]
            actors=day.events['actor'][selected]
            objects=day.events['object'][selected]
            reverse=np.isin(day.events['rel'][selected],reversed_ids)
            src=np.where(reverse,objects,actors)
            dst=np.where(reverse,actors,objects)
            nodes,encoded=np.unique(np.r_[src,dst],return_inverse=True)
            edge=np.unique(encoded[:len(src)]*len(nodes)+encoded[len(src):])
            if len(nodes)<2 or len(edge)==0:continue
            yield nodes,kinds[nodes],np.stack((edge//len(nodes),edge%len(nodes)))


def tensors(graph,device):
    _,kinds,edges=graph
    return F.one_hot(torch.from_numpy(kinds.astype(np.int64)-1).to(device),num_classes=3).float(),torch.from_numpy(edges).to(device)


def process_embeddings(model,graphs_,device,active=None):
    with torch.no_grad():
        for graph in graphs_:
            ids,kinds,_=graph
            keep=kinds==1
            if active is not None:keep &= np.isin(ids,active)
            if not keep.any():continue
            features,edges=tensors(graph,device)
            vectors=model.encoder(features,edges)['h'][keep].cpu().numpy()
            yield ids[keep],vectors


def score(model,graphs_,device,reference,active):
    scores={}
    graph_count=0
    for ids,vectors in process_embeddings(model,graphs_,device,active):
        graph_count+=1
        for offset in range(0,len(ids),256):
            q=torch.from_numpy(vectors[offset:offset+256]).to(device)
            distance=torch.cdist(q,reference)
            values=distance.topk(min(5,len(reference)),largest=False).values.mean(1).cpu().numpy()
            for actor,value in zip(ids[offset:offset+256],values):
                scores[int(actor)]=max(scores.get(int(actor),0.),float(value))
    return scores,graph_count


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--cap',type=int,default=8192)
    parser.add_argument('--reference-size',type=int,default=8192)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--sanity',action='store_true')
    args=parser.parse_args()
    started=time.monotonic()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise RuntimeError('Refusing to overwrite result')
    torch.set_num_threads(2);seed_everything(args.seed)
    device=torch.device(args.device)
    split=json.loads((args.cache/'split.json').read_text())
    source=ROOT/('data/processed/'+split['dataset']+'_e3')
    schema=json.loads((source/'schema.json').read_text())
    reversed_ids=[int(key) for key,value in schema['relation_names'].items() if value in REVERSED]
    kinds=np.load(source/'node_types.npy',mmap_mode='r')
    train_rows=np.load(args.cache/'train_targets.npy')
    train_graphs=[]
    for date_index,date in enumerate(split['train_dates']):
        day=EventDay(source,date)
        train_graphs.extend(graphs(day,train_rows[train_rows[:,0]==date_index,1],reversed_ids,args.cap,kinds))
        day.close()
    if args.sanity:train_graphs=train_graphs[:3];args.epochs=1
    model=GraphModel().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0005)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    history=[]
    for epoch in range(args.epochs):
        model.train()
        losses=[]
        for index in np.random.RandomState(args.seed+epoch).permutation(len(train_graphs)):
            features,edges=tensors(train_graphs[index],device)
            optimizer.zero_grad()
            loss=model.loss(features,edges)
            if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite graph loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        row={'epoch':epoch+1,'mean_loss':float(np.mean(losses)),'graphs':len(losses)}
        history.append(row);print(json.dumps(row),flush=True)
    train_seconds=time.monotonic()-started
    model.eval()
    torch.save(dict(model=model.state_dict(),dataset=split['dataset'],seed=args.seed),out/'checkpoint.pt')
    learned=[q for _,q in process_embeddings(model,iter(train_graphs),device)]
    vectors=np.concatenate(learned,axis=0)
    selected=np.random.RandomState(args.seed).choice(len(vectors),min(len(vectors),args.reference_size),replace=False)
    reference=torch.from_numpy(vectors[selected]).to(device)
    np.save(out/'reference.npy',vectors[selected])
    cohort=json.loads((args.cache/'cohort.json').read_text())
    views={}
    conditions=['validation'] if args.sanity else ['validation','test_clean','test_global_relation','test_global_time']
    for condition in conditions:
        start=time.monotonic()
        date=split['validation_date'] if condition=='validation' else split['test_date']
        day=EventDay(source,date)
        indices=np.load(args.cache/condition/'indices.npy',mmap_mode='r')
        if args.sanity:indices=indices[:8192]
        active=np.unique(day.events['actor'][indices])
        scores,count=score(model,graphs(day,indices,reversed_ids,args.cap,kinds),device,reference,active)
        day.close()
        if condition=='validation':
            alpha=.001 if len(scores)>=10000 else .01
            threshold=float(np.quantile(list(scores.values()),1-alpha))
        else:
            metrics=process_metrics(cohort['universe'],cohort['positive_processes'],scores,threshold=threshold)
            views[condition[5:]]=dict(metrics=metrics,graphs=count,events=len(indices),seconds=time.monotonic()-start)
            (out/(condition+'_process_scores.json')).write_text(json.dumps(scores),encoding='utf-8')
            print(json.dumps(views[condition[5:]]),flush=True)
    result=dict(status='SANITY_PASSED' if args.sanity else 'COMPLETED',method='MAGIC-adapted (PIDSMaker)',
                upstream_commit=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),
                config={**vars(args),'cache':str(args.cache),'output':str(out)},
                adaptations=['type-only adapter for legacy PyG GATConv signature',
                             'same fixed UTC cohorts and sampled benign train targets','15-minute snapshots capped by chronological raw-edge shards',
                             'fuse duplicate directed endpoints within shard','type-only node input',
                             'fixed 12 optimization epochs and benign process embedding reference with cap',
                             'mean 5-NN distance and max across graph shards; clean-validation threshold; no online adaptation',
                             'actor-only scored universe shared with sequence baselines; within-shard retrospective context'],
                views=views,history=history,threshold=threshold,training_seconds=train_seconds,total_seconds=time.monotonic()-started,
                reference_embeddings=len(reference),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
                rss_end_mib=psutil.Process().memory_info().rss/2**20)
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
