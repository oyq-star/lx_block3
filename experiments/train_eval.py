"""Train the planned small GRU and evaluate fixed, real-data process cohorts."""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn
import torch.nn.functional as F
from causal_data import EventDay,draw_contexts,encode_positions
from evaluation import process_metrics

ROOT=Path(__file__).resolve().parents[1]

class EventGRU(nn.Module):
    def __init__(self,vocabulary,embedding=32,hidden=64):
        super().__init__();self.embedding=nn.Embedding(vocabulary,embedding,padding_idx=0)
        self.gru=nn.GRU(embedding,hidden,batch_first=True);self.head=nn.Linear(hidden,vocabulary)
    def forward(self,x,lengths):
        z,_=self.gru(self.embedding(x))
        last=z[torch.arange(len(x),device=x.device),lengths-1]
        return self.head(last)

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def padded(sequences,k):
    x=np.zeros((len(sequences),k),dtype=np.int64);length=np.zeros(len(sequences),dtype=np.int64)
    for j,s in enumerate(sequences):x[j,:len(s)]=s;length[j]=len(s)
    return x,length

def cache_arrays(folder):
    folder=Path(folder)
    if not (folder/'CACHE_COMPLETE').exists():raise RuntimeError('Cache incomplete: '+str(folder))
    return {name:np.load(folder/(name+'.npy'),mmap_mode='r') for name in ['contexts','lengths','targets','actors','seq','indices']}

def tensor_batch(cache,indices,device):
    x=torch.from_numpy(np.array(cache['contexts'][indices],dtype=np.int64)).to(device)
    lens=torch.from_numpy(np.array(cache['lengths'][indices],dtype=np.int64)).to(device)
    y=torch.from_numpy(np.array(cache['targets'][indices],dtype=np.int64)).to(device)
    return x,lens,y

def check_deadline(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Declared run deadline reached')

def validation_ce(model,cache,rows,device,batch,deadline=float('inf')):
    model.eval();total=0.;count=0
    with torch.no_grad():
        for a in range(0,len(rows),batch):
            check_deadline(deadline)
            x,l,y=tensor_batch(cache,rows[a:a+batch],device)
            total+=float(F.cross_entropy(model(x,l),y,reduction='sum').item());count+=len(y)
    return total/max(1,count)

def score_cache(model,cache,device,batch,path,deadline=float('inf')):
    n=len(cache['targets']);output=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,))
    scores={};model.eval();started=time.monotonic();last=started;batch_ms=[]
    max_rss=psutil.Process().memory_info().rss
    with torch.no_grad():
        for a in range(0,n,batch):
            check_deadline(deadline)
            batch_start=time.perf_counter()
            b=min(a+batch,n)
            contexts=cache['contexts'][a:b];lengths=cache['lengths'][a:b];targets=cache['targets'][a:b]
            changes=np.r_[True,(lengths[1:]!=lengths[:-1]) | (targets[1:]!=targets[:-1]) |
                         np.any(contexts[1:]!=contexts[:-1],axis=1)]
            rows=a+np.flatnonzero(changes);x,l,y=tensor_batch(cache,rows,device)
            distinct_loss=F.cross_entropy(model(x,l),y,reduction='none').cpu().numpy()
            loss=distinct_loss[np.cumsum(changes)-1]
            if not np.isfinite(loss).all():raise RuntimeError('Non-finite real-data NLL')
            output[a:a+len(loss)]=loss
            ids=cache['actors'][a:a+len(loss)]
            boundaries=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1]
            for actor,value in zip(ids[boundaries],np.maximum.reduceat(loss,boundaries)):
                actor=int(actor);scores[actor]=max(scores.get(actor,-np.inf),float(value))
            batch_ms.append(1000*(time.perf_counter()-batch_start))
            if (a//batch)%64==0:max_rss=max(max_rss,psutil.Process().memory_info().rss)
            if time.monotonic()-last>20:
                print(json.dumps(dict(stage='score',path=path.name,rows=a,total=n,elapsed_s=round(time.monotonic()-started))),flush=True);last=time.monotonic()
    output.flush()
    elapsed=time.monotonic()-started
    return scores,dict(rows=n,seconds=elapsed,mean_CE=float(np.mean(output)) if n else None,
                       events_per_second=n/elapsed,batch_latency_ms_p50_p95=np.quantile(batch_ms,[.5,.95]).tolist() if batch_ms else [],
                       rss_peak_mib=max_rss/2**20)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--mode',choices=['clean','iid','dispersed','block','crop'],required=True)
    ap.add_argument('--seed',type=int,default=17);ap.add_argument('--mask-seed',type=int,default=23)
    ap.add_argument('--embedding',type=int,default=32);ap.add_argument('--hidden',type=int,default=64)
    ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--eval-batch-size',type=int,default=256)
    ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--patience',type=int,default=2)
    ap.add_argument('--learning-rate',type=float,default=.001);ap.add_argument('--clip',type=float,default=5.)
    ap.add_argument('--train-mask-durations',default='30,120');ap.add_argument('--clean-probability',type=float,default=.2)
    ap.add_argument('--device',default='cuda');ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--max-seconds',type=int,default=1800);ap.add_argument('--sanity',action='store_true')
    ap.add_argument('--views',default=None)
    ap.add_argument('--train-bank',type=Path)
    args=ap.parse_args();start=time.monotonic();deadline=start+args.max_seconds
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise RuntimeError('Refusing to overwrite completed results')
    torch.set_num_threads(args.threads);seed_everything(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested local CUDA unavailable')
    device=torch.device(args.device);cache=Path(args.cache)
    dataset=json.loads((cache/'dataset.json').read_text());split=dataset['split']
    root=ROOT/('data/processed/'+split['dataset']+'_e3')
    token_map=np.load(cache/'token_map.npy')
    rows=np.load(cache/'train_targets.npy')
    days=[] if args.train_bank else [EventDay(root,d,split['context'],split['lookback_seconds']) for d in split['train_dates']]
    val=cache_arrays(cache/'validation');val_original=np.load(cache/'validation_targets.npy')
    val_rows=np.searchsorted(val['indices'],val_original)
    if not np.array_equal(val['indices'][val_rows],val_original):raise RuntimeError('Validation sample alignment')
    if args.sanity:rows=rows[:256];val_rows=val_rows[:256];args.epochs=1
    model=EventGRU(dataset['vocabulary_size'],args.embedding,args.hidden).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    durations=tuple(int(x) for x in args.train_mask_durations.split(','))
    initial_val=validation_ce(model,val,val_rows,device,args.eval_batch_size,deadline)
    best=float('inf');stale=0;history=[];counts={'K_raw':0,'K_eff':0,'empty':0,'degenerate':0,'targets':0};last=start
    max_rss=psutil.Process().memory_info().rss
    config=dict(vars(args),dataset=dataset,parameters=sum(p.numel() for p in model.parameters()),
                torch=torch.__version__,cuda=torch.version.cuda,initial_validation_CE=initial_val,
                label_source='ORTHRUS/PIDSMaker, not model-generated labels',
                regime='fixed predeclared two-hour real-data pilot; target-conditional contexts')
    config['train_bank']=str(args.train_bank) if args.train_bank else None
    config['optimization_budget']='fixed epochs and identical targets; patience is diagnostic, never variant-specific stopping'
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    with (out/'train.jsonl').open('w',encoding='utf-8') as log:
        for epoch in range(args.epochs):
            bank=None
            if args.train_bank:
                bank=np.load(args.train_bank/(str(epoch)+'.npz'))
                clean_x,clean_l,aug_x,aug_l,labels=(bank[key] for key in ('clean','lengths','aug','aug_lengths','labels'))
                for key in counts:counts[key]+=int(bank[key]) if not args.sanity else 0
                bank.close()
            model.train();order=np.random.RandomState(args.seed+epoch*101).permutation(len(rows));total=0.;n=0
            for begin in range(0,len(order),args.batch_size):
                if time.monotonic()-start>args.max_seconds:raise TimeoutError('Run exceeded declared wall-time budget')
                indices=order[begin:begin+args.batch_size]
                if bank is not None:
                    cx,cl,ax,al,ys=clean_x[indices],clean_l[indices],aug_x[indices],aug_l[indices],labels[indices]
                else:
                    clean=[];aug=[];ys=[]
                    for di,i in rows[indices]:
                        day=days[int(di)];i=int(i)
                        cp,ap_,summary=draw_contexts(day,i,args.mode,args.mask_seed,epoch,durations,args.clean_probability)
                        clean.append(encode_positions(day,cp,token_map));aug.append(encode_positions(day,ap_,token_map))
                        ys.append(token_map[int(day.events['rel'][i])*4+int(day.events['otype'][i])])
                        for key in ['K_raw','K_eff','empty','degenerate']:counts[key]+=summary[key]
                        counts['targets']+=1
                    cx,cl=padded(clean,split['context']);ax,al=padded(aug,split['context'])
                both_x=torch.from_numpy(np.concatenate((cx,ax)).astype(np.int64,copy=False)).to(device)
                both_l=torch.from_numpy(np.concatenate((cl,al)).astype(np.int64,copy=False)).to(device)
                y=torch.from_numpy(np.asarray(ys,dtype=np.int64)).to(device)
                optimizer.zero_grad()
                clean_logits,aug_logits=model(both_x,both_l).split(len(ys))
                loss=.5*F.cross_entropy(clean_logits,y)+.5*F.cross_entropy(aug_logits,y)
                if not torch.isfinite(loss).item():raise RuntimeError('Non-finite training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),args.clip);optimizer.step()
                total+=float(loss.item())*len(ys);n+=len(ys)
                max_rss=max(max_rss,psutil.Process().memory_info().rss)
                if time.monotonic()-last>20:
                    entry=dict(epoch=epoch+1,targets=n,mean_loss=total/n,seconds=time.monotonic()-start,
                               rss_MiB=max_rss/2**20)
                    log.write(json.dumps(entry)+'\n');log.flush();print(json.dumps(entry),flush=True);last=time.monotonic()
            ce=validation_ce(model,val,val_rows,device,args.eval_batch_size,deadline)
            entry=dict(epoch=epoch+1,mean_training_loss=total/max(1,n),validation_CE=ce,targets=n)
            history.append(entry);log.write(json.dumps(entry)+'\n');log.flush();print(json.dumps(entry),flush=True)
            if ce<best:
                best=ce;stale=0;torch.save(dict(model=model.state_dict(),epoch=epoch+1,config=config),out/'best.pt')
            else:stale+=1
            # Best-checkpoint selection is independent of the fixed optimizer-step budget.
    training_seconds=time.monotonic()-start
    model.load_state_dict(torch.load(out/'best.pt',map_location=device)['model'])
    if args.sanity:
        result=dict(status='SANITY_PASSED',efficacy_evidence=False,history=history,training_seconds=training_seconds,
                    initial_validation_CE=initial_val,best_validation_CE=best,
                    parameters=config['parameters'],torch_cuda_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
                    process_RSS_peak_MiB=max_rss/2**20,device=str(device),real_data_targets=len(rows))
        (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True);return
    vs,vinfo=score_cache(model,val,device,args.eval_batch_size,out/'validation_nll.npy',deadline)
    alpha=.001 if len(vs)>=10000 else .01
    threshold=float(np.quantile(list(vs.values()),1-alpha))
    cohort=json.loads((cache/'cohort.json').read_text());results={}
    folders=[cache/('test_'+view) for view in args.views.split(',')] if args.views else sorted(cache.glob('test_*'))
    for folder in folders:
        if time.monotonic()-start>args.max_seconds:raise TimeoutError('Evaluation exceeded run budget')
        view=folder.name[5:];arrays=cache_arrays(folder)
        scores,info=score_cache(model,arrays,device,args.eval_batch_size,out/(view+'_event_nll.npy'),deadline)
        metrics=process_metrics(cohort['universe'],cohort['positive_processes'],scores,threshold=threshold)
        results[view]=dict(metrics=metrics,inference=info)
        with (out/(view+'_process_predictions.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['process_id','true_label','score'])
            for actor in cohort['universe']:writer.writerow([actor,int(actor in cohort['positive_processes']),scores.get(actor,'')])
    if device.type=='cuda':torch.cuda.synchronize()
    result=dict(status='COMPLETED',efficacy_scope='initial fixed-window pilot, one seed',mode=args.mode,seed=args.seed,
                training_seconds=training_seconds,total_wall_seconds=time.monotonic()-start,history=history,
                validation=dict(best_CE=best,alpha=alpha,threshold=threshold,scoreable_processes=len(vs),inference=vinfo),
                training_corruption=counts,views=results,cohort=cohort,
                resources=dict(torch_cuda_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
                               torch_cuda_reserved_peak_MiB=torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else None,
                               process_RSS_peak_MiB=max([max_rss/2**20,psutil.Process().memory_info().rss/2**20,vinfo['rss_peak_mib']]+[row['inference']['rss_peak_mib'] for row in results.values()]),
                               process_gpu_memory_MiB=None,
                               gpu_memory_note='PyTorch allocator values exclude driver and other processes'))
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],mode=args.mode,seconds=result['total_wall_seconds'],views={k:v['metrics'] for k,v in results.items()})),flush=True)

if __name__=='__main__':main()
