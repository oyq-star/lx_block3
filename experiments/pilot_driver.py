"""One local GPU, serial jobs, aggregate wall budget including scoring and sanity."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--modes',default='clean');ap.add_argument('--budget-seconds',type=int,default=7200)
    ap.add_argument('--run-seconds',type=int,default=1800);ap.add_argument('--train-mask-durations',default='1790,1799')
    ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--eval-batch-size',type=int,default=256)
    ap.add_argument('--sanity-only',action='store_true');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    state_path=out/'pilot_state.json'
    if state_path.exists():raise RuntimeError('Use a fresh output folder; pilot state exists')
    start=time.monotonic();limit=min(args.budget_seconds,10800)
    state=dict(status='RUNNING',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               aggregate_budget_seconds=limit,one_gpu=True,backend='local',jobs=[],gpu_wall_seconds=0)
    def save():state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    jobs=[('sanity','clean',True)]
    if not args.sanity_only:jobs +=[(mode,mode,False) for mode in args.modes.split(',')]
    env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for name,mode,sanity in jobs:
        remaining=limit-(time.monotonic()-start)
        if remaining<30:state['status']='BUDGET_STOP';break
        run_limit=min(args.run_seconds,int(remaining))
        folder=out/name;folder.mkdir(exist_ok=True)
        command=[sys.executable,str(ROOT/'experiments/train_eval.py'),'--cache',args.cache,'--output',str(folder),
                 '--mode',mode,'--epochs',str(args.epochs),'--max-seconds',str(max(1,run_limit-5)),
                 '--train-mask-durations',args.train_mask_durations,'--eval-batch-size',str(args.eval_batch_size)]
        if sanity:command.append('--sanity')
        job=dict(name=name,mode=mode,sanity=sanity,command=command,status='RUNNING',timeout_seconds=run_limit)
        state['jobs'].append(job);save();job_start=time.monotonic()
        print(json.dumps(dict(stage='launch',name=name,command=command)),flush=True)
        with (folder/'console.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=str(ROOT))
            last=job_start
            while process.poll() is None:
                now=time.monotonic()
                if now-job_start>=run_limit or now-start>=limit:
                    process.kill();process.wait();job['status']='TIMEOUT';break
                if now-last>=20:
                    print(json.dumps(dict(stage='monitor',name=name,seconds=round(now-job_start),
                                          aggregate_seconds=round(now-start))),flush=True);last=now
                time.sleep(1)
        job['seconds']=time.monotonic()-job_start;job['returncode']=process.returncode
        state['gpu_wall_seconds']+=job['seconds']
        result_path=folder/'results.json'
        if result_path.exists():
            result=json.loads(result_path.read_text());job['result_status']=result['status']
        if job['status']!='TIMEOUT':job['status']='COMPLETED' if process.returncode==0 else 'FAILED'
        save();print(json.dumps(job),flush=True)
        if job['status']!='COMPLETED':state['status']=job['status'];break
    else:state['status']='COMPLETED'
    state['total_seconds']=time.monotonic()-start;save();print(json.dumps(state),flush=True)
if __name__=='__main__':main()
