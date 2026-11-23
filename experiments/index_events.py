"""Index full UTC days by process and event time, including causal midnight warmup."""
import argparse
import datetime as dt
import json
import time
from pathlib import Path
import numpy as np
import psutil
from prepare_dataset import EVENT_DTYPE,DAY_NS

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['theia','cadets'])
    p.add_argument('--dates',default='all');p.add_argument('--lookback-seconds',type=int,default=1800)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    folder=root/('data/processed/'+args.dataset+'_e3')
    if not (folder/'EXTRACTION_COMPLETE').exists():raise RuntimeError('Extraction not complete')
    schema=json.loads((folder/'schema.json').read_text())
    dates=sorted(schema['days']) if args.dates=='all' else args.dates.split(',')
    for date in dates:
        target=folder/('day_'+date);target.mkdir(exist_ok=True)
        if (target/'INDEX_COMPLETE').exists():continue
        start=time.monotonic();day=dt.datetime.strptime(date,'%Y-%m-%d').replace(tzinfo=dt.timezone.utc)
        start_ns=int(day.timestamp())*10**9;end_ns=start_ns+DAY_NS
        path=folder/('events_'+date+'.bin')
        raw=np.memmap(path,mode='r',dtype=EVENT_DTYPE)
        previous=folder/('events_'+(day-dt.timedelta(days=1)).strftime('%Y-%m-%d')+'.bin')
        warm=None
        if previous.exists() and previous.stat().st_size:
            prev=np.memmap(previous,mode='r',dtype=EVENT_DTYPE)
            warm=prev[(prev['t']>=start_ns-args.lookback_seconds*10**9)&(prev['t']<start_ns)]
            del prev
        n_warm=0 if warm is None else len(warm)
        joined=np.lib.format.open_memmap(target/'unsorted.npy',mode='w+',dtype=EVENT_DTYPE,shape=(len(raw)+n_warm,))
        for a in range(0,len(raw),100000):
            b=min(a+100000,len(raw));joined[a:b]=raw[a:b]
        if n_warm:joined[len(raw):]=warm
        joined.flush();del raw,warm
        # Original event UUID is the stable tie breaker, original database _id resolves duplicates.
        order=np.lexsort((joined['seq'],joined['uid'],joined['t'],joined['actor']))
        events=np.lib.format.open_memmap(target/'events.npy',mode='w+',dtype=EVENT_DTYPE,shape=joined.shape)
        for a in range(0,len(order),100000):events[a:a+100000]=joined[order[a:a+100000]]
        events.flush();del joined,order
        for field in ('t','actor','object','rel','otype','seq'):
            field_array=np.lib.format.open_memmap(target/(field+'.npy'),mode='w+',dtype=events.dtype[field],shape=(len(events),))
            for a in range(0,len(events),100000):field_array[a:a+100000]=events[field][a:a+100000]
            field_array.flush();del field_array
        actor=events['actor'];changes=np.r_[0,np.flatnonzero(actor[1:]!=actor[:-1])+1,len(actor)]
        starts=np.full(schema['max_node_index']+1,-1,dtype=np.int64)
        ends=np.full_like(starts,-1)
        starts[actor[changes[:-1]]]=changes[:-1];ends[actor[changes[:-1]]]=changes[1:]
        np.save(target/'starts.npy',starts);np.save(target/'ends.npy',ends)
        targets=np.flatnonzero((events['t']>=start_ns)&(events['t']<end_ns))
        np.save(target/'targets.npy',targets)
        for rel in np.unique(events['rel']):
            np.save(target/('rel_'+str(int(rel))+'.npy'),np.flatnonzero(events['rel']==rel))
        info=dict(date_utc=date,events=len(events),target_events=len(targets),warmup_events=n_warm,
                  processes=int(np.count_nonzero(starts>=0)),elapsed_seconds=time.monotonic()-start,
                  stable_order=['process_index','timestamp_ns','original_event_uuid','database_row_id'],
                  rss_MiB=psutil.Process().memory_info().rss/2**20,
                  lookback_seconds=args.lookback_seconds)
        (target/'metadata.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
        (target/'INDEX_COMPLETE').write_text('complete\n')
        print(json.dumps(info),flush=True)
        del events,starts,ends,targets,actor

if __name__=='__main__':main()
