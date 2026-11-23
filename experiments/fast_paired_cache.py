"""Accelerate full-cohort paired caches by detecting identity masks in compiled batches."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from numba import njit
from causal_data import EventDay,draw_contexts,encode_positions
from build_caches import mmap_array


ROOT=Path(__file__).resolve().parents[1]
MODES=('iid','dispersed','block')


@njit
def screen(indices,times,actor,starts,seq,rel,positions,offsets,relations,k,lookback,mask_seed,duration):
    result=np.zeros((len(indices),2),dtype=np.int64)
    active=np.empty(len(relations),dtype=np.int32)
    for row,index in enumerate(indices):
        start=starts[actor[index]]
        target=times[index]
        prefix=times[start:index]
        lo=start+np.searchsorted(prefix,target-lookback)
        hi=start+np.searchsorted(prefix,target)
        if hi==lo:continue
        np.random.seed((mask_seed*1000003+seq[index]*9176+7919)%(2**32-1))
        np.random.random()
        count=0
        for key in range(len(relations)):
            pos=positions[offsets[key]:offsets[key+1]]
            if np.searchsorted(pos,hi)>np.searchsorted(pos,lo):
                active[count]=key;count+=1
        pick=np.random.randint(count) if count>1 else 0
        key=active[pick]
        left=int(target-lookback+np.random.random()*max(1,lookback-duration))
        right=min(left+duration,target)
        start_cut=lo+np.searchsorted(times[lo:hi],left)
        end_cut=lo+np.searchsorted(times[lo:hi],right)
        pos=positions[offsets[key]:offsets[key+1]]
        raw=np.searchsorted(pos,end_cut)-np.searchsorted(pos,start_cut)
        recent=max(lo,hi-k,start_cut)
        eff=np.searchsorted(pos,end_cut)-np.searchsorted(pos,recent) if end_cut>recent else 0
        result[row,0]=raw;result[row,1]=eff
    return result


def relation_data(day):
    positions=np.concatenate([day.relpos[rel] for rel in day.relations])
    offsets=np.r_[0,np.cumsum([len(day.relpos[rel]) for rel in day.relations])]
    return positions,offsets,np.array(day.relations,dtype=np.int32)


def masks(day,indices,relations,seed,duration):
    return screen(indices,day.events['t'],day.events['actor'],day.starts,day.events['seq'],day.events['rel'],
                  *relations,day.k,day.lookback,seed,duration*10**9)


def check_screen(day,indices,relations,seed,duration):
    sample=indices[np.random.RandomState(73).choice(len(indices),min(4096,len(indices)),replace=False)]
    expected=masks(day,sample,relations,seed,duration)
    for row,index in enumerate(sample):
        _,_,info=draw_contexts(day,int(index),'block',seed,0,(duration,),0)
        if tuple(expected[row])!=(info['K_raw'],info['K_eff']):
            raise RuntimeError(f'Mask differs from reference at {int(index)}: {expected[row]}, {info}')
    print(json.dumps(dict(stage='reference_match',duration=duration,targets=len(sample))),flush=True)


def prepare(cache,modes,duration,check_only=False):
    start=time.monotonic()
    split=json.loads((cache/'split.json').read_text())
    source=ROOT/('data/processed/'+split['dataset']+'_e3')
    mapping=np.load(cache/'token_map.npy')
    seed=split['mask_seed']
    clean=cache/'test_clean'
    indices=np.load(clean/'indices.npy',mmap_mode='r')
    day=EventDay(source,split['test_date'],split['context'],split['lookback_seconds'])
    relations=relation_data(day)
    check_screen(day,indices,relations,seed,duration)
    if check_only:return
    for mode in modes:
        end_name='unseen300' if duration==300 else mode
        folder=cache/('test_'+end_name)
        if (folder/'CACHE_COMPLETE').exists():continue
        folder.mkdir(parents=True,exist_ok=True)
        for name in ('indices','targets','actors','seq'):
            np.save(folder/(name+'.npy'),np.load(clean/(name+'.npy'),mmap_mode='r'))
        shape=(len(indices),day.k)
        contexts=mmap_array(folder/'contexts.npy',np.uint16,shape)
        lengths=mmap_array(folder/'lengths.npy',np.uint16,(len(indices),))
        original=np.load(clean/'contexts.npy',mmap_mode='r')
        original_lengths=np.load(clean/'lengths.npy',mmap_mode='r')
        total=np.zeros(2,dtype=np.int64);empty=degenerate=refill=0
        last=time.monotonic()
        for a in range(0,len(indices),8192):
            b=min(a+8192,len(indices))
            subset=indices[a:b]
            values=masks(day,subset,relations,seed,duration)
            total+=values.sum(axis=0)
            active=np.flatnonzero(values[:,1])
            empty+=b-a-len(active)
            contexts[a:b]=original[a:b];lengths[a:b]=original_lengths[a:b]
            for row in active:
                cp,ap,summary=draw_contexts(day,int(subset[row]),mode,seed,0,(duration,),0)
                if (summary['K_raw'],summary['K_eff'])!=tuple(values[row]):raise RuntimeError('Active count mismatch')
                input_tokens=encode_positions(day,ap,mapping)
                contexts[a+row]=0;contexts[a+row,:len(input_tokens)]=input_tokens
                lengths[a+row]=len(input_tokens)
                degenerate+=summary['degenerate'];refill+=int((ap<cp[0]).sum())
            if time.monotonic()-last>=20:
                print(json.dumps(dict(stage='paired_cache',mode=end_name,rows=b,total=len(indices))),flush=True)
                last=time.monotonic()
        contexts.flush();lengths.flush()
        metadata=dict(mode=mode,regime='target-conditional fixed cohort',date=day.date,rows=len(indices),context=day.k,
                      mask_seed=seed,duration=duration,elapsed_seconds=time.monotonic()-start,
                      corruption_summary=dict(K_raw=int(total[0]),K_eff=int(total[1]),empty=empty,degenerate=degenerate,refill_tokens=refill))
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        (folder/'CACHE_COMPLETE').write_text('complete\n')
        print(json.dumps(metadata),flush=True)
    day.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--modes',default=','.join(MODES))
    parser.add_argument('--duration',type=int,default=1799)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    prepare(args.cache,args.modes.split(','),args.duration,args.check_only)
