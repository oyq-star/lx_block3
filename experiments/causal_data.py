"""Past-only contexts, exact corruption controls and memory-mapped daily data."""
import json
from pathlib import Path
import numpy as np

NS=10**9

class EventDay:
    def __init__(self,root,date,context=128,lookback_seconds=1800):
        self.root=Path(root);self.date=date;self.k=context;self.lookback=lookback_seconds*NS
        folder=self.root/('day_'+date)
        if not (folder/'INDEX_COMPLETE').exists():raise RuntimeError('Day not indexed: '+str(folder))
        self.events={field:np.load(folder/(field+'.npy'),mmap_mode='r')
                     for field in ('t','actor','object','rel','otype','seq')}
        self.starts=np.load(folder/'starts.npy',mmap_mode='r')
        self.ends=np.load(folder/'ends.npy',mmap_mode='r')
        self.targets=np.load(folder/'targets.npy',mmap_mode='r')
        self.relpos={int(p.stem.split('_')[1]):np.load(p,mmap_mode='r') for p in folder.glob('rel_*.npy')}
        self.relations=sorted(self.relpos)

    def close(self):
        # Windows does not permit removing files while memmaps retain handles.
        for array in list(self.events.values())+[self.starts,self.ends,self.targets]+list(self.relpos.values()):
            mmap=getattr(array,'_mmap',None)
            if mmap is not None and not mmap.closed:mmap.close()

    def bounds(self,i):
        actor=int(self.events['actor'][i]);start=int(self.starts[actor]);t=int(self.events['t'][i])
        lo=start+int(np.searchsorted(self.events['t'][start:i],t-self.lookback,side='left'))
        # Strict timestamp exclusion: earlier tie-breaker records at exactly t are not past.
        hi=start+int(np.searchsorted(self.events['t'][start:i],t,side='left'))
        return lo,hi

    def count_rel(self,r,lo,hi):
        pos=self.relpos[r];return int(np.searchsorted(pos,hi)-np.searchsorted(pos,lo))

    def clean_positions(self,i):
        lo,hi=self.bounds(i);return np.arange(max(lo,hi-self.k),hi,dtype=np.int64)

    def raw_tokens(self,positions):
        return self.events['rel'][positions].astype(np.int32)*4+self.events['otype'][positions].astype(np.int32)

    def target_range(self,start_ns,end_ns):
        idx=self.targets
        t=self.events['t'][idx]
        return np.asarray(idx[(t>=start_ns)&(t<end_ns)])

def seeded_rng(seed,seq,epoch=0,salt=0):
    value=(int(seed)*1000003+int(seq)*9176+int(epoch)*10000019+salt*7919)%(2**32-1)
    return np.random.RandomState(value)

def merge_tail(parts,k):
    return np.sort(np.concatenate(parts))[-k:]

def other_tail(day,r,lo,hi,k):
    if k==0 or lo==hi:return np.empty(0,dtype=np.int64)
    start=max(lo,hi-max(2*k,256))
    tail=np.arange(start,hi,dtype=np.int64)
    keep=tail[day.events['rel'][start:hi]!=r][-k:]
    if len(keep)==k or start==lo:return keep
    parts=[keep]
    for relation,positions in day.relpos.items():
        if relation!=r:
            begin,end=np.searchsorted(positions,[lo,start])
            parts.append(positions[max(begin,end-k):end])
    return merge_tail(parts,k)

def uniform_tail(size,survivors,k,rng):
    """Largest k members of a uniform fixed-size subset, in ascending order."""
    k=min(k,survivors)
    if not k:return np.empty(0,dtype=np.int64)
    if size==survivors:return np.arange(size-k,size,dtype=np.int64)
    if size<=4096:return np.sort(rng.choice(size,survivors,replace=False))[-k:]
    if survivors<=1024:
        return np.sort(rng.choice(size,survivors,replace=False))[-k:] if size<=16384 else sparse_tail(size,survivors,k,rng)
    most_recent=min(size,max(256,4*k))
    if survivors*most_recent >= 2*k*size:
        count=rng.hypergeometric(survivors,size-survivors,most_recent)
        recent=np.sort(rng.choice(most_recent,count,replace=False))[-k:]+size-most_recent
        if count>=k:return recent
        return np.r_[sparse_tail(size-most_recent,survivors-count,k-count,rng),recent]
    return sparse_tail(size,survivors,k,rng)

def sparse_tail(size,survivors,k,rng):
    found=np.empty(k,dtype=np.int64)
    for offset in range(k):
        if survivors==size:
            found[offset:]=np.arange(size-1,size-1-(k-offset),-1)
            break
        gap=rng.binomial(size-survivors,rng.beta(1,survivors))
        size-=gap+1;survivors-=1
        found[offset]=size
    return found[::-1]

def draw_contexts(day,i,mode,mask_seed=23,epoch=0,durations=(30,120),clean_probability=0.2):
    if any(d<=0 or d*NS>day.lookback for d in durations):raise ValueError('Duration outside lookback')
    lo,hi=day.bounds(i);wstart=max(lo,hi-day.k)
    clean=np.arange(wstart,hi,dtype=np.int64)
    summary={'K_raw':0,'K_eff':0,'prefix_events':hi-lo,'empty':1,'degenerate':0}
    if hi==lo or mode=='clean':return clean,clean,summary
    seq=int(day.events['seq'][i]);rng=seeded_rng(mask_seed,seq,epoch,1)
    if rng.rand()<clean_probability:return clean,clean,summary
    ranges={r:np.searchsorted(pos,[lo,hi]) for r,pos in day.relpos.items()}
    active=[r for r in day.relations if ranges[r][0]<ranges[r][1]]
    r=active[int(rng.randint(len(active)))];duration=int(durations[int(rng.randint(len(durations)))])*NS
    t=int(day.events['t'][i]);lower=t-day.lookback
    a=int(lower+rng.random_sample()*max(1,day.lookback-duration));b=min(a+duration,t)
    # Timestamp interval is mapped within this process before the target.
    times=day.events['t'][lo:hi]
    aidx=lo+int(np.searchsorted(times,a));bidx=lo+int(np.searchsorted(times,b))
    kraw=day.count_rel(r,aidx,bidx)
    removed=(day.events['rel'][clean]==r)&(day.events['t'][clean]>=a)&(day.events['t'][clean]<b)
    keff=int(removed.sum());summary.update(K_raw=kraw,K_eff=keff,empty=int(keff==0),relation=r,
                                       duration_seconds=duration//NS)
    if keff==0:return clean,clean,summary
    if mode=='block':
        kept=clean[~removed];need=day.k-len(kept)
        past=np.arange(max(lo,min(aidx,wstart)-need),min(aidx,wstart))
        recent=np.arange(max(lo,bidx,wstart-need),wstart)
        middle=other_tail(day,r,aidx,min(bidx,wstart),need) if aidx<min(bidx,wstart) else np.empty(0,dtype=np.int64)
        return clean,merge_tail((kept,past,recent,middle),day.k),summary
    if mode not in {'iid','dispersed','crop'}:raise ValueError(mode)
    control=seeded_rng(mask_seed,seq,epoch,2)
    if mode=='crop':
        count=min(day.k,max(1,hi-lo-kraw))
        start=int(control.randint(lo,max(lo+1,hi-count+1)))
        return clean,np.arange(start,start+count,dtype=np.int64),summary
    if mode=='iid':candidates=np.arange(len(clean));outside_n=wstart-lo
    else:
        candidates=np.flatnonzero(day.events['rel'][clean]==r)
        outside_n=day.count_rel(r,lo,wstart)
    selected=control.choice(candidates,size=keff,replace=False)
    inside_keep=np.ones(len(clean),dtype=bool);inside_keep[selected]=False
    kept=clean[inside_keep];need=day.k-len(kept)
    samples=uniform_tail(outside_n,outside_n-(kraw-keff),need,control)
    if mode=='iid':refill=samples+lo
    else:
        pos=day.relpos[r];begin=ranges[r][0]
        refill=merge_tail((pos[begin+samples],other_tail(day,r,lo,wstart,need)),need)
    summary['degenerate']=int(keff==len(candidates) and (outside_n==0 or kraw-keff==outside_n))
    augmented=np.r_[refill,kept]
    actual_eff=len(np.setdiff1d(clean,augmented,assume_unique=True))
    if actual_eff!=keff:raise RuntimeError('K_eff mismatch')
    return clean,augmented,summary

def encode_positions(day,positions,token_map):
    if len(positions)==0:return np.array([1],dtype=np.uint16) # START
    return token_map[day.raw_tokens(positions)].astype(np.uint16)

def positive_processes(root,date):
    labels=json.loads((Path(root)/'labels.json').read_text())['process_labels']
    return {int(k) for k,rows in labels.items() if any(x['date_utc']==date for x in rows)}
