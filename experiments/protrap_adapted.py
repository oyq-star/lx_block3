"""Bounded PRO-TRAP adaptation: upstream edge GAT and temporal graph contrast."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import dgl
import numpy as np
import psutil
import torch
import torch.nn.functional as F

from causal_data import EventDay, NS
from evaluation import process_metrics
from prepare_dataset import REVERSED
from train_eval import seed_everything


ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'external/Pro-Trap'


def shards(day, indices, reversed_ids, kinds, cap):
    times=day.events['t'][indices]
    order=np.argsort(times,kind='stable')
    ordered=indices[order]
    buckets=times[order]//(900*NS)
    boundaries=np.r_[0,np.flatnonzero(buckets[1:]!=buckets[:-1])+1,len(ordered)]
    for first,last in zip(boundaries[:-1],boundaries[1:]):
        for offset in range(first,last,cap):
            ids=ordered[offset:min(last,offset+cap)]
            actors=day.events['actor'][ids]
            objects=day.events['object'][ids]
            relations=day.events['rel'][ids].astype(np.int64)
            reverse=np.isin(relations,reversed_ids)
            src=np.where(reverse,objects,actors)
            dst=np.where(reverse,actors,objects)
            nodes,encoded=np.unique(np.r_[src,dst],return_inverse=True)
            if len(nodes)<2:continue
            yield nodes,kinds[nodes],encoded[:len(ids)],encoded[len(ids):],relations,day.events['t'][ids]


def graph(shard,dim,mask=None):
    nodes,kinds,src,dst,relations,_=shard
    if mask is not None:src,dst,relations=src[mask],dst[mask],relations[mask]
    result=dgl.graph((src,dst),num_nodes=len(nodes))
    result.ndata['attr']=F.one_hot(torch.from_numpy(kinds.astype(np.int64)-1),3).float()
    result.edata['attr']=F.one_hot(torch.from_numpy(relations.astype(np.int64,copy=False)),dim).float()
    return result


def two_hop_means(hidden,src,dst,seeds):
    adjacent={}
    for parent,child in zip(src.tolist(),dst.tolist()):
        adjacent.setdefault(parent,set()).add(child)
        adjacent.setdefault(child,set()).add(parent)
    pooled=[]
    for node in seeds:
        first=adjacent[node]
        neighbors={node}|first
        for child in first:neighbors.update(adjacent[child])
        pooled.append(hidden[sorted(neighbors)].mean(dim=0))
    return torch.stack(pooled)


def contrast(model,shard,dim,rng):
    _,_,src,dst,_,times=shard
    relative=(times-times[0])/NS
    span=float(relative[-1])
    if not span:return None
    offset=rng.uniform(0,span/3)
    starts=offset+np.arange(3)*span/6
    masks=[(relative>=left)&(relative<=left+span/3) for left in starts]
    common=set.intersection(*(set(src[mask].tolist()) for mask in masks))
    if len(common)<2:return None
    seeds=rng.choice(sorted(common),min(len(common),128),replace=False).tolist()
    pooled=[two_hop_means(model.embed(graph(shard,dim,mask)),src[mask],dst[mask],seeds) for mask in masks]
    target=torch.arange(len(seeds))
    terms=[]
    for left in range(len(pooled)):
        for right in range(left+1,len(pooled)):
            dot=pooled[left]@pooled[right].T
            terms.append(.5*(F.cross_entropy(dot,target)+F.cross_entropy(dot.T,target)))
    return .6*torch.stack(terms).sum()


def embeddings(model,shards_,dim,active=None):
    with torch.no_grad():
        for item in shards_:
            ids,kinds,*_=item
            keep=kinds==1
            if active is not None:keep &= np.isin(ids,active)
            if keep.any():yield ids[keep],model.embed(graph(item,dim))[keep]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--cap',type=int,default=4096)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--sanity',action='store_true')
    args=parser.parse_args()
    start=time.monotonic()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise RuntimeError('Refusing to overwrite result')
    sys.path.insert(0,str(UPSTREAM))
    from model.autoencoder_cl import build_model
    torch.set_num_threads(2);seed_everything(args.seed)
    split=json.loads((args.cache/'split.json').read_text())
    source=ROOT/('data/processed/'+split['dataset']+'_e3')
    schema=json.loads((source/'schema.json').read_text())
    reversed_ids=[int(key) for key,value in schema['relation_names'].items() if value in REVERSED]
    mapping=np.load(args.cache/'token_map.npy')
    known={int(raw)//4 for raw in np.flatnonzero(mapping!=2)}
    relations={value:index+1 for index,value in enumerate(sorted(known))}
    relation_lookup=np.array([relations.get(index,0) for index in range(len(schema['relation_names']))])
    dimension=len(known)+1
    kinds=np.load(source/'node_types.npy',mmap_mode='r')
    train_rows=np.load(args.cache/'train_targets.npy')
    train_shards=[]
    for date_index,date in enumerate(split['train_dates']):
        day=EventDay(source,date)
        train_shards.extend(shards(day,train_rows[train_rows[:,0]==date_index,1],reversed_ids,kinds,args.cap))
        day.close()
    def remap(items):
        for nodes,types,src,dst,edge_type,times in items:yield nodes,types,src,dst,relation_lookup[edge_type],times
    train_shards=list(remap(train_shards))
    model=build_model(SimpleNamespace(num_hidden=64,num_layers=3,negative_slope=.2,mask_rate=.5,
                                      alpha_l=3,n_dim=3,e_dim=dimension,sr=0))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0005)
    history=[]
    for epoch in range(1 if args.sanity else args.epochs):
        model.train()
        rng=np.random.RandomState(args.seed+epoch)
        losses=[]
        for idx in rng.permutation(len(train_shards)):
            optimizer.zero_grad()
            loss=contrast(model,train_shards[idx],dimension,rng)
            if loss is None:continue
            if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite contrastive loss')
            loss.backward();optimizer.step();losses.append(float(loss.item()))
            if args.sanity and len(losses)>=3:break
        if not losses:raise RuntimeError('No shared contrastive seeds')
        row=dict(epoch=epoch+1,mean_loss=float(np.mean(losses)),optimized_shards=len(losses),
                 candidate_shards=len(train_shards))
        history.append(row);print(json.dumps(row),flush=True)
    training_seconds=time.monotonic()-start
    model.eval()
    torch.save(model.state_dict(),out/'checkpoint.pt')
    reference=torch.cat([vector for _,vector in embeddings(model,train_shards,dimension)],dim=0)
    if len(reference)>8192:
        chosen=np.random.RandomState(args.seed).choice(len(reference),8192,replace=False)
        reference=reference[chosen]
    np.save(out/'reference.npy',reference.numpy())
    cohort=json.loads((args.cache/'cohort.json').read_text())
    views={}
    for condition in (['validation'] if args.sanity else ['validation','test_clean','test_global_relation','test_global_time']):
        before=time.monotonic()
        date=split['validation_date'] if condition=='validation' else split['test_date']
        day=EventDay(source,date)
        ids=np.load(args.cache/condition/'indices.npy',mmap_mode='r')
        if args.sanity:ids=ids[:4096]
        active=np.unique(day.events['actor'][ids])
        scores={}
        for nodes,vectors in embeddings(model,remap(shards(day,ids,reversed_ids,kinds,args.cap)),dimension,active):
            for offset in range(0,len(nodes),256):
                distances=torch.cdist(vectors[offset:offset+256],reference)
                values=distances.topk(min(5,len(reference)),largest=False).values.mean(1).numpy()
                for node,value in zip(nodes[offset:offset+256],values):
                    scores[int(node)]=max(scores.get(int(node),0.),float(value))
        day.close()
        if condition=='validation':
            alpha=.001 if len(scores)>=10000 else .01
            threshold=float(np.quantile(list(scores.values()),1-alpha))
        else:
            result=process_metrics(cohort['universe'],cohort['positive_processes'],scores,threshold=threshold)
            views[condition[5:]]=dict(metrics=result,events=len(ids),seconds=time.monotonic()-before)
            (out/(condition+'_process_scores.json')).write_text(json.dumps(scores),encoding='utf-8')
            print(json.dumps(views[condition[5:]]),flush=True)
    info=dict(status='SANITY_PASSED' if args.sanity else 'COMPLETED',method='PRO-TRAP-adapted',
              upstream_commit=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),
              seed=args.seed,device='cpu',dgl=dgl.__version__,epochs_completed=len(history),cap=args.cap,history=history,
              adaptations=['upstream entity-level edge GAT and default subgraph-only contrast direction (gcl=1,ncl=0,sr=0)',
                           'three mid-overlap views; full bounded view embeddings + two-hop pooling replace DGL neighbor loader',
                           'same benign train sample and clean/shared-stream test cohorts, chronological shards',
                           'train-only relation one-hot with UNK; 12 epochs instead of upstream 50',
                           'fixed benign process reference, mean 5-NN and max process, frozen validation threshold'],
              views=views,threshold=threshold,training_seconds=training_seconds,
              total_seconds=time.monotonic()-start,rss_end_mib=psutil.Process().memory_info().rss/2**20)
    (out/'results.json').write_text(json.dumps(info,indent=2),encoding='utf-8')


if __name__=='__main__':main()
