"""Stream a PIDSMaker public PostgreSQL dump into compact causal event arrays.

No database server, full graph, model output, or test label is used in parsing.
Node labels are author annotations derived from the DARPA ground-truth report.
"""
import argparse
import csv
import datetime as dt
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
PG=ROOT/'tools/postgresql/pgsql/bin/pg_restore.exe'
EVENT_DTYPE=np.dtype([('t','<i8'),('actor','<i4'),('object','<i4'),('rel','<i2'),
                      ('otype','u1'),('uid','S36'),('seq','<i8')])
REVERSED={'EVENT_EXECUTE','EVENT_LSEEK','EVENT_MMAP','EVENT_OPEN','EVENT_ACCEPT',
          'EVENT_READ','EVENT_RECVFROM','EVENT_RECVMSG','EVENT_READ_SOCKET_PARAMS',
          'EVENT_CHECK_FILE_ATTRIBUTES','READ'}
DAY_NS=86400*10**9

def copy_rows(dump,table,log_dir):
    err=log_dir/(table+'.stderr.log')
    with err.open('wb') as error:
        p=subprocess.Popen([str(PG),'--data-only','--file=-','--table='+table,str(dump)],
                           stdout=subprocess.PIPE,stderr=error,bufsize=2**20)
        inside=False
        for line in p.stdout:
            if not inside:
                if line.startswith(('COPY public.'+table+' ').encode()):inside=True
                continue
            if line.rstrip(b'\r\n')==b'\\.':inside=False;continue
            if inside:yield line.rstrip(b'\r\n').split(b'\t')
        code=p.wait()
    if code:raise RuntimeError('%s failed (%s): %s'%(table,code,err.read_text(errors='replace')[-2000:]))

def load_annotations(dataset):
    folder=ROOT/'external/PIDSMaker/Ground_Truth/orthrus'/('E3-'+dataset.upper())
    annotations={}
    for p in sorted(folder.glob('*.csv')):
        if dataset=='theia':
            date='2018-04-10' if 'Firefox_Backdoor' in p.name else '2018-04-12'
        else:
            suffix=p.stem.rsplit('_',1)[-1]
            if suffix not in {'06','12','13'}:continue
            date='2018-04-'+suffix
        for line in p.read_text(encoding='utf-8').splitlines():
            uuid=line.split(',',1)[0].strip().upper()
            if len(uuid)==36 and "'subject':" in line:
                annotations.setdefault(uuid,[]).append({'campaign':p.stem,'date_utc':date})
    return annotations,folder

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['theia','cadets'],required=True)
    ap.add_argument('--chunk-events',type=int,default=100000)
    ap.add_argument('--skip-events',action='store_true')
    args=ap.parse_args();started=time.monotonic()
    dump=ROOT/('data/downloads/'+args.dataset+'_e3.dump')
    out=ROOT/('data/processed/'+args.dataset+'_e3');out.mkdir(parents=True,exist_ok=True)
    logs=out/'logs';logs.mkdir(exist_ok=True)
    schema=subprocess.run([str(PG),'--schema-only','--file=-',str(dump)],stdout=subprocess.PIPE,check=True).stdout
    (out/'schema.sql').write_bytes(schema)
    annotations,gt_folder=load_annotations(args.dataset)
    node_types=np.zeros(2**20,dtype=np.uint8)
    subject_uuids={};node_counts=Counter();max_id=-1
    with (out/'subjects.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['index_id','uuid','path','command'])
        for table,type_id in [('file_node_table',2),('netflow_node_table',3),('subject_node_table',1)]:
            for fields in copy_rows(dump,table,logs):
                index=int(fields[-1]);max_id=max(max_id,index)
                if index>=len(node_types):
                    size=max(index+1,len(node_types)*2);expanded=np.zeros(size,dtype=np.uint8)
                    expanded[:len(node_types)]=node_types;node_types=expanded
                if node_types[index] and node_types[index]!=type_id:
                    raise RuntimeError('Conflicting node type for index '+str(index))
                node_types[index]=type_id;node_counts[table]+=1
                if type_id==1:
                    uuid=fields[0].decode().upper();subject_uuids[index]=uuid
                    writer.writerow([index,uuid,fields[2].decode('utf-8','replace'),fields[3].decode('utf-8','replace')])
            print(json.dumps(dict(stage='nodes',table=table,count=node_counts[table],rss_MiB=psutil.Process().memory_info().rss/2**20)),flush=True)
    node_types=node_types[:max_id+1];np.save(out/'node_types.npy',node_types)
    labels={str(i):annotations[u] for i,u in subject_uuids.items() if u in annotations}
    (out/'labels.json').write_text(json.dumps(dict(source='ORTHRUS/PIDSMaker node annotations derived from DARPA report',
             source_folder=str(gt_folder),annotated_process_uuids=len(annotations),
             mapped_processes=len(labels),unmapped_annotated_uuids=sorted(set(annotations)-set(subject_uuids.values())),
             process_labels=labels),indent=2),encoding='utf-8')
    del subject_uuids
    stats=dict(dataset=args.dataset,host=args.dataset+'_e3_host',timestamp_unit='nanoseconds since Unix epoch',
               evaluation_timezone='UTC',node_counts=dict(node_counts),max_node_index=max_id,
               label_source='author-curated ORTHRUS node annotations; not official event labels',
               dtype=EVENT_DTYPE.descr,relation_names={},days={},dropped=Counter(),records=0,
               mapped_attack_processes=len(labels),start_wall_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    if args.skip_events:
        (out/'schema.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');return
    relations={};rows=[];handles={};day_counts=Counter();seen=0;last_report=time.monotonic()
    def flush():
        if not rows:return
        block=np.array(rows,dtype=EVENT_DTYPE);rows.clear()
        days=block['t']//DAY_NS
        for day in np.unique(days):
            if int(day) not in handles:
                date=dt.datetime.fromtimestamp(int(day)*86400,dt.timezone.utc).strftime('%Y-%m-%d')
                file=out/('events_'+date+'.bin')
                if file.exists():raise RuntimeError('Refusing to append to prior extraction: '+str(file))
                handles[int(day)]=file.open('wb')
            selected=block[days==day];selected.tofile(handles[int(day)]);day_counts[int(day)]+=len(selected)
    try:
        for fields in copy_rows(dump,'event_table',logs):
            seen+=1
            if len(fields)!=8:
                raise RuntimeError('Unexpected COPY field count '+str(len(fields)))
            src,dst=int(fields[1]),int(fields[4]);op=fields[2].decode()
            if src<0 or dst<0 or src>=len(node_types) or dst>=len(node_types):
                stats['dropped']['unknown_node_index']+=1;continue
            actor,obj=(dst,src) if op in REVERSED else (src,dst)
            if node_types[actor]!=1:
                stats['dropped']['reconstructed_actor_not_subject']+=1;continue
            if node_types[obj]==0:
                stats['dropped']['unknown_object_type']+=1;continue
            if op not in relations:relations[op]=len(relations)
            rows.append((int(fields[6]),actor,obj,relations[op],int(node_types[obj]),fields[5],int(fields[7])))
            if len(rows)>=args.chunk_events:flush()
            if time.monotonic()-last_report>20:
                print(json.dumps(dict(stage='events',seen=seen,kept=sum(day_counts.values())+len(rows),elapsed_s=round(time.monotonic()-started),rss_MiB=round(psutil.Process().memory_info().rss/2**20))),flush=True)
                last_report=time.monotonic()
        flush()
    finally:
        for f in handles.values():f.close()
    stats['records']=seen;stats['kept_events']=sum(day_counts.values())
    stats['relation_names']={str(v):k for k,v in relations.items()}
    stats['days']={dt.datetime.fromtimestamp(k*86400,dt.timezone.utc).strftime('%Y-%m-%d'):v for k,v in sorted(day_counts.items())}
    stats['elapsed_seconds']=time.monotonic()-started
    stats['rss_MiB']=psutil.Process().memory_info().rss/2**20
    (out/'schema.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    (out/'EXTRACTION_COMPLETE').write_text('complete\n',encoding='ascii')
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()
