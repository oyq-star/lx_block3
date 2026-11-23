"""Process-level metrics against author ground truth, including unobservable positives."""
import math
import numpy as np

def process_metrics(universe,positive_ids,scores,budgets=(10,50,100),threshold=None):
    ids=np.asarray(sorted(set(int(x) for x in universe)),dtype=np.int64)
    truth=np.isin(ids,np.asarray(sorted(set(int(x) for x in positive_ids)),dtype=np.int64))
    values=np.asarray([scores.get(int(x),np.nan) for x in ids],dtype=float)
    observed=np.isfinite(values);positive_total=int(truth.sum())
    positive_observed=int(truth[observed].sum())
    oid=ids[observed];oval=values[observed];oy=truth[observed]
    order=np.lexsort((oid,-oval));oval=oval[order];oy=oy[order];oid=oid[order]
    ap_all=0.0 if positive_total else None
    if positive_total and positive_observed:
        end=np.r_[np.flatnonzero(oval[1:]!=oval[:-1])+1,len(oval)]
        cum=np.cumsum(oy);prior=0
        for j in end:
            hits=int(cum[j-1])-prior
            ap_all+=(hits/positive_total)*(float(cum[j-1])/int(j));prior=int(cum[j-1])
    result=dict(processes=len(ids),positive_processes=positive_total,
                scoreable_processes=int(observed.sum()),positive_scoreable=positive_observed,
                positive_unscored=positive_total-positive_observed,
                positive_unobservable=positive_total-positive_observed, # legacy name; includes object-only, unscored entities
                positive_coverage=positive_observed/positive_total if positive_total else None,
                AP_all=ap_all,
                AP_observed=ap_all*positive_total/positive_observed if positive_observed else None)
    for budget in budgets:
        n=min(int(budget),len(oy));tp=int(oy[:n].sum())
        boundary_ties=int(np.sum(oval==oval[n-1])) if n else 0
        result['top_'+str(budget)]=dict(alerts=n,tp=tp,fp=n-tp,boundary_tied_processes=boundary_ties,
            recall=tp/positive_total if positive_total else None,precision=tp/n if n else None)
    if threshold is not None and math.isfinite(float(threshold)):
        alarm=observed&(values>=threshold);tp=int(truth[alarm].sum());fp=int((~truth[alarm]).sum())
        result['frozen_threshold']=dict(threshold=float(threshold),tp=tp,fp=fp,fn=positive_total-tp,
             recall=tp/positive_total if positive_total else None,precision=tp/(tp+fp) if tp+fp else None)
    return result
