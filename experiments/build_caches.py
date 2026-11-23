"""Freeze causal pilot splits, vocabulary, sampled training targets and eval contexts."""
import argparse
import datetime as dt
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import psutil
from causal_data import EventDay,draw_contexts,encode_positions,positive_processes,NS

ROOT=Path(__file__).resolve().parents[1]

def timestamp(date,hour):
    return int(dt.datetime.fromisoformat(date+'T'+hour+':00+00:00').timestamp())*NS

def mmap_array(path,dtype,shape):
    return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)

def write_eval(day,target_indices,token_map,dest,mode,mask_seed,duration,clean_probability=0):
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'CACHE_COMPLETE').exists():
        saved=json.loads((dest/'metadata.json').read_text())
        for key,value in dict(mode=mode,date=day.date,rows=len(target_indices),context=day.k,
                              mask_seed=mask_seed,duration=duration).items():
            if saved.get(key)!=value:raise RuntimeError('Incompatible completed cache: '+key)
        return
    n=len(target_indices);context=mmap_array(dest/'contexts.npy',np.uint16,(n,day.k))
    np.save(dest/'indices.npy',target_indices)
    length=mmap_array(dest/'lengths.npy',np.uint16,(n,))
    target=mmap_array(dest/'targets.npy',np.uint16,(n,))
    actors=mmap_array(dest/'actors.npy',np.int32,(n,))
    seq=mmap_array(dest/'seq.npy',np.int64,(n,))
    counts=Counter();start=time.monotonic();last=start
    for row,i in enumerate(target_indices):
        if mode=='clean':positions=day.clean_positions(int(i));summary={}
        else:
            _,positions,summary=draw_contexts(day,int(i),mode,mask_seed,0,(duration,),clean_probability)
        encoded=encode_positions(day,positions,token_map)
        context[row,:]=0;context[row,:len(encoded)]=encoded;length[row]=len(encoded)
        raw=int(day.events['rel'][i])*4+int(day.events['otype'][i])
        target[row]=token_map[raw];actors[row]=day.events['actor'][i];seq[row]=day.events['seq'][i]
        for key in ('K_raw','K_eff','empty','degenerate'):counts[key]+=summary.get(key,0)
        if time.monotonic()-last>20:
            print(json.dumps(dict(stage='eval_cache',mode=mode,date=day.date,row=row,total=n,elapsed_s=round(time.monotonic()-start))),flush=True);last=time.monotonic()
    for array in (context,length,target,actors,seq):array.flush()
    metadata=dict(mode=mode,regime='target-conditional, fixed targets, not full-stream outage',
                  date=day.date,rows=n,context=day.k,mask_seed=mask_seed,duration=duration,
                  corruption_summary=dict(counts),elapsed_seconds=time.monotonic()-start,
                  rss_MiB=psutil.Process().memory_info().rss/2**20)
    (dest/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (dest/'CACHE_COMPLETE').write_text('complete\n')
    print(json.dumps(metadata),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['theia','cadets'])
    ap.add_argument('--train-per-day',type=int,default=10000);ap.add_argument('--validation-targets',type=int,default=4096)
    ap.add_argument('--context',type=int,default=128);ap.add_argument('--lookback-seconds',type=int,default=1800)
    ap.add_argument('--seed',type=int,default=17);ap.add_argument('--mask-seed',type=int,default=23)
    ap.add_argument('--duration',type=int,default=120);ap.add_argument('--views',default='clean,iid,dispersed,block')
    ap.add_argument('--cache-name',default=None)
    args=ap.parse_args();root=ROOT/('data/processed/'+args.dataset+'_e3')
    schema=json.loads((root/'schema.json').read_text());available=set(schema['days'])
    if args.dataset=='theia':
        train=[x for x in sorted(available) if '2018-04-02'<=x<='2018-04-08']
        val='2018-04-09';test='2018-04-10';begin='18:00';end='20:00'
    else:
        train=[x for x in ['2018-04-03','2018-04-04'] if x in available]
        val='2018-04-05';test='2018-04-06';begin='15:00';end='17:00'
    for x in train+[val,test]:
        if x not in available:raise RuntimeError('Declared split missing date '+x)
    dest=ROOT/'data/cache'/(args.cache_name or args.dataset+'_pilot');dest.mkdir(parents=True,exist_ok=True)
    split=dict(dataset=args.dataset,timezone='UTC',train_dates=train,validation_date=val,
               test_date=test,test_start_utc=test+'T'+begin+':00Z',test_end_utc=test+'T'+end+':00Z',
               validation_start_utc=val+'T'+begin+':00Z',validation_end_utc=val+'T'+end+':00Z',
               rationale='fixed two-hour pilot window covering the complete published campaign window plus benign margins; no performance-based window selection',
               annotation_source='ORTHRUS/PIDSMaker annotations derived from DARPA report',
               is_full_benchmark=False,seed=args.seed,mask_seed=args.mask_seed,
               context=args.context,lookback_seconds=args.lookback_seconds,duration_seconds=args.duration)
    if (dest/'split.json').exists() and json.loads((dest/'split.json').read_text())!=split:
        raise RuntimeError('Existing cache split differs; choose a fresh --cache-name')
    (dest/'split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    rng=np.random.RandomState(args.seed);vocab=set();train_rows=[]
    for di,date in enumerate(train):
        day=EventDay(root,date,args.context,args.lookback_seconds)
        # Training-only vocabulary; no val/test vocabulary or label affects fitting.
        ids=day.targets
        for a in range(0,len(ids),100000):vocab.update(day.raw_tokens(ids[a:a+100000]).tolist())
        sample=np.sort(rng.choice(ids,size=min(args.train_per_day,len(ids)),replace=False))
        train_rows.extend((di,int(i)) for i in sample)
        day.close();del day
    max_raw=max(int(x) for x in schema['relation_names'])*4+4
    token_map=np.full(max_raw+1,2,dtype=np.uint16)
    for i,raw in enumerate(sorted(vocab),3):token_map[raw]=i
    np.save(dest/'token_map.npy',token_map)
    np.save(dest/'train_targets.npy',np.array(train_rows,dtype=np.int64))
    vd=EventDay(root,val,args.context,args.lookback_seconds)
    vtarget=vd.target_range(timestamp(val,begin),timestamp(val,end))
    if not len(vtarget):raise RuntimeError('Empty predeclared benign validation window')
    sample=np.sort(rng.choice(vtarget,size=min(args.validation_targets,len(vtarget)),replace=False))
    np.save(dest/'validation_targets.npy',sample)
    write_eval(vd,vtarget,token_map,dest/'validation','clean',args.mask_seed,args.duration)
    vd.close()
    day=EventDay(root,test,args.context,args.lookback_seconds)
    target=day.target_range(timestamp(test,begin),timestamp(test,end))
    if not len(target):raise RuntimeError('Empty predeclared test window')
    # Universe includes process entities in either role; object-only processes cannot be silently omitted.
    actor=np.asarray(day.events['actor'][target]);obj=day.events['object'][target]
    universe=np.unique(np.r_[actor,obj[day.events['otype'][target]==1]])
    positives=positive_processes(root,test)&set(universe.tolist())
    cohort=dict(universe=universe.tolist(),positive_processes=sorted(positives),
                all_annotated_for_day=sorted(positive_processes(root,test)),
                annotated_outside_cohort=sorted(positive_processes(root,test)-set(universe.tolist())),
                original_target_events=len(target),date=test,partial_day_hours=2,
                budget_pool='fixed two-hour pilot cohort, not extrapolated to full host-day')
    (dest/'cohort.json').write_text(json.dumps(cohort,indent=2),encoding='utf-8')
    config=dict(split=split,train_targets=len(train_rows),validation_targets=len(sample),
                vocabulary_size=3+len(vocab),tokens_seen_in_training=sorted(vocab),
                train_sampling='uniform without replacement within each available declared training day',
                target_events=len(target),processes=len(universe),positive_processes=len(positives))
    (dest/'dataset.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for mode in args.views.split(','):
        write_eval(day,target,token_map,dest/('test_'+mode),mode,args.mask_seed,args.duration)
    print(json.dumps(config,indent=2),flush=True)

if __name__=='__main__':main()
