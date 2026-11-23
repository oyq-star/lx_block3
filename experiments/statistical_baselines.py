"""Marginal and first-order transition baselines fitted on the same real train targets."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from causal_data import EventDay,encode_positions
from evaluation import process_metrics

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--views',default=None)
    ap.add_argument('--smoothing',type=float,default=1.);args=ap.parse_args();start=time.monotonic()
    cache=Path(args.cache);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=json.loads((cache/'dataset.json').read_text());split=config['split'];v=config['vocabulary_size']
    root=ROOT/('data/processed/'+split['dataset']+'_e3');mapping=np.load(cache/'token_map.npy')
    days=[EventDay(root,d,split['context'],split['lookback_seconds']) for d in split['train_dates']]
    marginal=np.full(v,args.smoothing);transition=np.full((v,v),args.smoothing)
    for di,i in np.load(cache/'train_targets.npy'):
        day=days[int(di)];i=int(i);context=day.clean_positions(i)
        last=int(encode_positions(day,context,mapping)[-1])
        target=int(mapping[int(day.events['rel'][i])*4+int(day.events['otype'][i])])
        marginal[target]+=1;transition[last,target]+=1
    marginal/=marginal.sum();transition/=transition.sum(1,keepdims=True)
    cohort=json.loads((cache/'cohort.json').read_text());results={}
    for method in ['frequency','markov']:
        method_results={};threshold=None
        test=[cache/('test_'+view) for view in args.views.split(',')] if args.views else sorted(cache.glob('test_*'))
        for folder in [cache/'validation']+test:
            x=np.load(folder/'contexts.npy',mmap_mode='r');length=np.load(folder/'lengths.npy',mmap_mode='r')
            targets=np.load(folder/'targets.npy',mmap_mode='r');actors=np.load(folder/'actors.npy',mmap_mode='r')
            scores={};n=len(targets)
            event_nll=np.lib.format.open_memmap(out/(method+'_'+folder.name+'_nll.npy'),mode='w+',dtype=np.float32,shape=(n,))
            for a in range(0,n,65536):
                b=min(a+65536,n);ys=targets[a:b].astype(np.int64)
                if method=='frequency':nll=-np.log(marginal[ys])
                else:
                    prev=x[np.arange(a,b),length[a:b].astype(np.int64)-1].astype(np.int64)
                    nll=-np.log(transition[prev,ys])
                event_nll[a:b]=nll
                for actor,value in zip(actors[a:b],nll):
                    actor=int(actor);scores[actor]=max(scores.get(actor,-np.inf),float(value))
            event_nll.flush()
            if folder.name=='validation':
                alpha=.001 if len(scores)>=10000 else .01;threshold=float(np.quantile(list(scores.values()),1-alpha))
            else:method_results[folder.name[5:]]=process_metrics(cohort['universe'],cohort['positive_processes'],scores,threshold=threshold)
        results[method]=dict(threshold=threshold,views=method_results)
    info=dict(status='COMPLETED',model_family='statistics',real_data=True,scope='fixed-window one-seed pilot',
              training_targets=config['train_targets'],seconds=time.monotonic()-start,results=results)
    (out/'results.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    np.savez(out/'fitted_probabilities.npz',marginal=marginal,transition=transition)
    print(json.dumps(info),flush=True)

if __name__=='__main__':main()
