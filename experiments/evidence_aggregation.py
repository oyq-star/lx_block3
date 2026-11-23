"""Matched evidence analysis and a label-free alert-budget selector."""
import argparse
from contextlib import closing
import csv
from itertools import combinations

import joblib
import numpy as np
from scipy.stats import ks_2samp, spearmanr, rankdata
from sklearn.base import clone

from add_stress_views import utc_ns
from causal_data import EventDay, positive_processes
from evaluation import process_metrics
from method_followup import (Case, OUTPUT as PRIOR, MODES, feature_groups, forest_inputs,
                             path_aggregates, scenarios, write_csv, fuse, path_scores)
from semantic_fusion import ROOT, RUN, fit_semantics, load, path_index, save_json, tail
from count_baseline import features as count_features
from structural_fusion import features as structure_features

OUTPUT = ROOT/'runs/evidence_aggregation_v1'
QS = (.95,.97,.99,.995)
RULES = ('maximum','max_median')
BRANCHES = ('Path','Count','Structure')
CHANNELS = ('image','file','creation')
SELECT = {'theia': [('14:30','15:00'),('17:00','17:30')],
          'cadets': [('06:00','08:00'),('14:00','15:00')]}
ADDITIONAL = {'theia':('2018-04-11','2018-04-13'), 'cadets':('2018-04-07','2018-04-08')}


def join_calibrated(matrix):
    counts = np.isfinite(matrix).sum(axis=1)
    sorted_scores = np.sort(np.nan_to_num(matrix,nan=-np.inf),axis=1)
    maximum = np.where(counts,sorted_scores[:,-1],0)
    mean = np.nansum(matrix,axis=1)/np.maximum(1,counts)
    top2 = np.where(counts>1,(maximum+sorted_scores[:,-2])*.5,maximum)
    return dict(max=maximum,top2=top2,mean=mean)


def path_grid(ref, value):
    paths, channels = {}, {}
    for mode in MODES:
        ref_raw, raw = ref['path'][mode], value['path'][mode]
        score = np.column_stack([tail(ref_raw[:,i], raw[:,i]) for i in range(len(CHANNELS))])
        available = np.isfinite(raw) & np.isfinite(ref_raw).any(axis=0)
        score = np.where(available,score,np.nan)
        channels[mode] = score
        for cross,result in join_calibrated(score).items():
            paths[(mode,cross)] = result
    return paths, channels


def branch_matrix(ref, value):
    paths, _ = path_grid(ref,value)
    return np.column_stack([paths['max','max'], *(tail(ref['forest'][branch],value['forest'][branch])
                                                  for branch in BRANCHES[1:])])


def qualifying_quantiles(x):
    return np.quantile(x,QS)


def alarm_rate(scores, limits):
    return np.mean(scores[:,None]>=limits,axis=0)


def score_rules(branches):
    f = fuse(*branches.T)
    return {rule:f[rule] for rule in RULES}


def intervals(scores, bins, repeats=1000):
    blocks = [scores[bins==k] for k in range(12)]
    rng = np.random.RandomState(27)
    clustered = []
    for sample in rng.randint(0,12,size=(repeats,12)):
        view = np.concatenate([blocks[i] for i in sample])
        if len(view):
            clustered.append(qualifying_quantiles(view))
    rng = np.random.RandomState(27)
    individual = [qualifying_quantiles(scores[rng.randint(0,len(scores),size=len(scores))])
                  for _ in range(repeats)]
    return np.quantile(clustered,[.05,.95],axis=0),np.quantile(individual,[.05,.95],axis=0)


def ranges(dataset, valdate):
    def cached(role):
        row = load(PRIOR/dataset/'benign'/role/'scores_complete.json')
        return row['date'], *row['window']
    return dict(reference=cached('reference'), threshold=cached('threshold'),
                **{f'select{i}':(valdate,*w) for i,w in enumerate(SELECT[dataset],1)},
                audit1=cached('audit'),
                **{f'audit{i}':(date,'18:00','20:00') for i,date in enumerate(ADDITIONAL[dataset],2)})


def corr(a,b):
    if len(a)<2 or a.min()==a.max() or b.min()==b.max():
        return None
    return float(spearmanr(a,b).correlation)


def diagnostics(prefix, matrix, groups):
    return [dict(**prefix,population=name,left=BRANCHES[i],right=BRANCHES[j],n=int(mask.sum()),
                 rho=corr(matrix[mask,i],matrix[mask,j]))
            for name,mask in groups.items() for i,j in combinations(range(3),2)]


def column_groups(definitions,n):
    return {name:np.atleast_1d(np.arange(n)[cols]) for name,cols in definitions.items()}


def redundancy(train, groups):
    for branch,matrix in train.items():
        indices = column_groups(groups[branch],matrix.shape[1])
        active = np.ptp(matrix,axis=0)>0
        finite = np.flatnonzero(active)
        association = np.full((matrix.shape[1],matrix.shape[1]),np.nan)
        association[np.ix_(finite,finite)] = np.corrcoef(np.apply_along_axis(rankdata,0,matrix[:,finite]),rowvar=False)
        for (a,A),(b,B) in combinations(indices.items(),2):
            rho = association[np.ix_(A,B)]
            if np.isfinite(rho).any():
                i,j = np.unravel_index(np.nanargmax(np.abs(rho)),rho.shape)
                yield dict(branch=branch,left=a,right=b,actors=len(matrix),left_column=int(A[i]),
                           right_column=int(B[j]),rho=rho[i,j])


def main(dataset):
    output = OUTPUT/dataset
    output.mkdir(exist_ok=True,parents=True)
    source = ROOT/'data/processed'/f'{dataset}_e3'
    cache = ROOT/'data/cache'/f'{dataset}_longoutage_v1'
    v1,v2 = (ROOT/f'runs/semantic_fusion_v{i}'/dataset for i in (1,2))
    split = load(cache/'split.json')
    mapping = np.load(cache/'token_map.npy')
    vocabulary = load(cache/'dataset.json')['vocabulary_size']
    schema = load(source/'schema.json')
    roles = [int(i) for i,n in schema['relation_names'].items() if n in {'EVENT_CLONE','EVENT_FORK'}]
    subjects,files = path_index(dataset,source,v1)
    support,actors = fit_semantics(source,split['train_dates'],subjects,files,roles)
    semantic_args = subjects,files,roles,support,actors
    forests = dict(Count=joblib.load(RUN/dataset/'isolation_forest/model.joblib'),
                   Structure=joblib.load(v2/'structure_forest.joblib'))

    def get_evidence(case,indices,include_features=False):
        with closing(EventDay(source,case.date)) as day:
            log_actors = day.events['actor'][indices]
            offsets = np.r_[0,np.flatnonzero(log_actors[1:]!=log_actors[:-1])+1]
            ids = log_actors[offsets]
            count_ids,count = count_features(log_actors,mapping[day.raw_tokens(indices)],vocabulary)
            struct_ids,structure = structure_features(day,indices,mapping,vocabulary)
            np.testing.assert_array_equal(ids,count_ids)
            np.testing.assert_array_equal(ids,struct_ids)
            local = structure_features(day,indices,mapping,vocabulary,degree_scope='actor')[1] if include_features else None
            if local is not None:
                np.testing.assert_array_equal(local[:,:-2],structure[:,:-2])
            t = day.events['t'][indices[offsets]]
        x = dict(Count=count,Structure=structure)
        return dict(actors=ids,events=np.diff(np.r_[offsets,len(indices)]),first=t,features=x,local=local,
                    forest={name:-model.score_samples(x[name]) for name,model in forests.items()},
                    path=path_aggregates(source,case,indices,semantic_args,ids))

    window_spec = ranges(dataset,split['validation_date'])
    benign,win_rows = {},[]
    for role,(date,a,b) in window_spec.items():
        if positive_processes(source,date):
            raise ValueError(f'Window {date} overlaps a day with positive annotations')
        with closing(EventDay(source,date)) as day:
            idx = day.target_range(utc_ns(date+'T'+a+':00Z'),utc_ns(date+'T'+b+':00Z'))
        v = get_evidence(Case(date,role,'unannotated',None,()),idx) if len(idx) else None
        if v is not None:
            benign[role]=v
        win_rows.append(dict(role=role,date=date,start=a,end=b,events=len(idx),actors=len(v['actors']) if v else 0))
        print(dataset,role,date,a,b,len(idx),flush=True)
    write_csv(output/'windows.csv',win_rows)
    ref = benign['reference']
    cut_bins = (benign['threshold']['first']-utc_ns(window_spec['threshold'][0]+'T'+window_spec['threshold'][1]+':00Z'))//(600*10**9)
    assert cut_bins.min()>=0 and cut_bins.max()<12
    split_matrix = {role:branch_matrix(ref,v) for role,v in benign.items()}
    matrices = {role:score_rules(scores) for role,scores in split_matrix.items()}
    cuts = {rule:qualifying_quantiles(scores) for rule,scores in matrices['threshold'].items()}
    choice_rows = []
    for rule in RULES:
        errors=[]
        for name in ('select1','select2'):
            for q,rate in zip(QS,alarm_rate(matrices[name][rule],cuts[rule])):
                error=abs(rate-(1-q))
                errors.append(error)
                choice_rows.append(dict(rule=rule,window=name,quantile=q,rate=rate,absolute_error=error))
    criterion = {rule:float(np.mean([r['absolute_error'] for r in choice_rows if r['rule']==rule])) for rule in RULES}
    selected=min(RULES,key=lambda r:criterion[r])
    save_json(output/'selector.json',dict(rule=selected,errors=criterion,nominal_quantiles=QS,
        exclusions='No annotated ranking labels or audit windows used; maximum wins exact ties.'))
    write_csv(output/'selector_rates.csv',choice_rows)
    audit_rows=[]
    for rule in RULES:
        threshold_scores = matrices['threshold'][rule]
        block,individual=intervals(threshold_scores,cut_bins)
        for name,window in matrices.items():
            scores=window[rule]
            ks=ks_2samp(threshold_scores,scores).statistic
            for i,q in enumerate(QS):
                cut=cuts[rule][i]
                alert=int(np.count_nonzero(scores>=cut))
                audit_rows.append(dict(rule=rule,selected=rule==selected,window=name,quantile=q,cutoff=cut,
                    actors=len(scores),alerts=alert,rate=alert/len(scores),ks_vs_threshold=ks,median=np.median(scores),
                    score_q99=np.quantile(scores,.99),block_low=block[0,i],block_high=block[1,i],
                    actor_low=individual[0,i],actor_high=individual[1,i]))
    for row in audit_rows:
        if row['rule']=='maximum' and row['window']=='audit1' and row['quantile']==.99:
            assert row['alerts']=={'theia':12,'cadets':27}[dataset],row
    write_csv(output/'threshold_transfer.csv',audit_rows)
    write_csv(output/'block_sizes.csv',[dict(bin=i,actors=int(np.count_nonzero(cut_bins==i))) for i in range(12)])

    train = forest_inputs(source,split,mapping,vocabulary)
    groups = feature_groups(vocabulary)
    write_csv(output/'feature_redundancy.csv',list(redundancy(train,groups)))
    fitted = {}
    for branch,features in train.items():
        cols = column_groups(groups[branch],features.shape[1])
        for group,idx in cols.items():
            other=np.setdiff1d(np.arange(features.shape[1]),idx)
            for kind,columns in [('only',idx),('drop',other)]:
                fitted[branch,kind,group]=(columns,clone(forests[branch]).fit(features[:,columns]))
    local_train=[]
    for date in split['train_dates']:
        with closing(EventDay(source,date)) as day:
            a,b=(utc_ns(date+split[name][10:]) for name in ('test_start_utc','test_end_utc'))
            indices=day.target_range(a,b)
            if len(indices):
                local_train.append(structure_features(day,indices,mapping,vocabulary,degree_scope='actor')[1])
    local_train=np.vstack(local_train)
    np.testing.assert_array_equal(local_train[:,:-2],train['Structure'][:,:-2])
    local_model=clone(forests['Structure']).fit(local_train)
    orig=Case(split['validation_date'],'original_reference','unannotated',cache/'validation',())
    original=get_evidence(orig,np.load(orig.folder/'indices.npy',mmap_mode='r'))
    activity_limits=np.quantile(ref['events'],[.25,.5,.75])
    if len(np.unique(activity_limits))!=3:
        raise ValueError(f'Degenerate activity quartiles: {activity_limits}')
    save_json(output/'activity_cuts.json',dict(quartiles=activity_limits.tolist(),ties='lower bin'))
    branch_limits=np.quantile(split_matrix['threshold'],.99,axis=0)
    correlations,features,ranking,activity,exceedance,coverage,channel_rows=[],[],[],[],[],[],[]

    def collect(prefix,v,matrix,groups_here):
        correlations.extend(diagnostics(prefix,matrix,groups_here))
        for name,mask in groups_here.items():
            for k in range(4):
                hist=(matrix[mask]>=branch_limits).sum(axis=1)
                exceedance.append(dict(**prefix,population=name,exceeding=k,n=int(mask.sum()),actors=int(np.sum(hist==k))))
        grid,channels=path_grid(ref,v)
        for j,channel in enumerate(CHANNELS):
            raw=v['path']['max'][:,j]
            calibrated=channels['max'][:,j]
            valid=np.isfinite(raw)
            multi=np.sum(np.isfinite(v['path']['max']),axis=1)>=2
            def winner(score):
                return int(np.count_nonzero(multi & np.isfinite(score[:,j]) &
                              (score[:,j]==np.max(np.nan_to_num(score,nan=-np.inf),axis=1))))
            channel_rows.append(dict(**prefix,channel=channel,actors=len(raw),present=int(valid.sum()),
                raw_q95=np.quantile(raw[valid],.95) if valid.any() else None,
                calibrated_q95=np.nanquantile(calibrated,.95) if np.isfinite(calibrated).any() else None,
                multi=int(multi.sum()),raw_max_count=winner(v['path']['max']),calibrated_max_count=winner(channels['max'])))
        if 'unannotated' in groups_here:
            mask=groups_here['unannotated']
            for (within,between),score in grid.items():
                activity.append(dict(**prefix,within=within,between=between,activity='all_unannotated',n=int(mask.sum()),
                    positives=None,AP=None,rho=corr(v['events'][mask],score[mask])))

    for name,v in benign.items():
        collect(dict(dataset=dataset,date=window_spec[name][0],condition=name,grid='unannotated'),v,split_matrix[name],
                dict(unannotated=np.ones(len(v['actors']),dtype=bool)))

    static={}
    for case in scenarios(dataset,split,cache):
        kind='fixed' if case.label in {'clean','iid','dispersed','block','unseen300'} else case.label
        key=case.date,kind
        if key not in static:
            static[key]=get_evidence(case,np.load(case.folder/'indices.npy',mmap_mode='r'),True)
        v=static[key]
        prefix=dict(dataset=dataset,date=case.date,condition=case.label,grid=case.group)
        ids=v['actors']
        positives=case.cohort['positive_processes']
        universe=case.cohort['universe']
        y=np.isin(ids,positives)
        def measure(s):
            return process_metrics(universe,positives,dict(zip(ids,s)),budgets=(50,))['AP_all']
        # AP grids use the original reference, matching the archived aggregation controls.
        path_variants,_=path_grid(original,v)
        old_paths=path_scores(original['path'],v['path'])
        for within in MODES:
            np.testing.assert_allclose(path_variants[within,'max'],old_paths[within],rtol=0,atol=0)
        strata=np.searchsorted(activity_limits,v['events'],side='left')
        for (within,between),score in path_variants.items():
            ranking.append(dict(**prefix,study='path',rule=within+'_'+between,AP=measure(score)))
            activity.append(dict(**prefix,within=within,between=between,activity='original_unannotated',
                n=int((~y).sum()),positives=None,AP=None,rho=corr(v['events'][~y],score[~y])))
            for level in range(4):
                keep=strata==level
                actors_here=ids[keep]
                true_here=ids[keep&y]
                ap=process_metrics(actors_here,true_here,dict(zip(actors_here,score[keep])),budgets=())['AP_all']
                activity.append(dict(**prefix,within=within,between=between,activity=str(level+1),n=len(actors_here),
                    positives=len(true_here),AP=ap,rho=None))
        split_branches=branch_matrix(ref,v)
        for rule,score in score_rules(split_branches).items():
            ranking.append(dict(**prefix,study='selector',rule=rule,AP=measure(score)))
        collect(prefix,v,split_branches,dict(annotated=y,unannotated=~y))
        top={name:set(np.lexsort((ids,-score))[:50]) for name,score in zip(BRANCHES,split_branches.T)}
        for left,right in [*( (x,x) for x in BRANCHES),*combinations(BRANCHES,2)]:
            hits=top[left]&top[right]
            coverage.append(dict(**prefix,left=left,right=right,intersect=len(hits),
                            positive_hits=int(y[list(hits)].sum()),scoreable_positives=int(y.sum())))
        coverage.append(dict(**prefix,left='union',right='union',intersect=len(set.union(*top.values())),
                             positive_hits=int(y[list(set.union(*top.values()))].sum()),scoreable_positives=int(y.sum())))
        for branch,score in v['forest'].items():
            features.append(dict(**prefix,branch=branch,mode='full',group='all',AP=measure(score)))
        features.append(dict(**prefix,branch='Structure',mode='local',group='both_degree_columns',AP=measure(-local_model.score_samples(v['local']))))
        for (branch,kind,group),(cols,model) in fitted.items():
            features.append(dict(**prefix,branch=branch,mode=kind,group=group,AP=measure(-model.score_samples(v['features'][branch][:,cols]))))
        print(dataset,case.date,case.label,flush=True)
    for name,rows in [('ranking',ranking),('feature_ablation',features),('activity',activity),('branch_correlation',correlations),
                      ('channel_scales',channel_rows),('branch_exceedance',exceedance),('top50_coverage',coverage)]:
        write_csv(output/(name+'.csv'),rows)
    with (PRIOR/dataset/'per_condition.csv').open(encoding='utf-8-sig',newline='') as stream:
        old=list(csv.DictReader(stream))
    control={}
    for r in old:
        if r['study']=='sequence' and r['rule']=='original_fusion' and r['variant'] in {'max','mean'}:
            control.setdefault((r['date'],r['view'],r['grid'],r['seed']),{})[r['variant']]=float(r['AP'])
    if not control:
        raise ValueError('No archived matched NLL controls')
    paired=[dict(dataset=dataset,date=date,condition=condition,grid=grid,seed=seed,AP_max=m['max'],
                 AP_mean=m['mean'],delta=m['mean']-m['max'])
            for (date,condition,grid,seed),m in control.items()]
    assert len(ranking)==11*len(paired)//3
    write_csv(output/'nll_pairs.csv',paired)
    save_json(output/'status.json',dict(status='COMPLETED',conditions=len(paired)//3,audits=sum(k.startswith('audit') for k in benign)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('theia','cadets'),required=True)
    main(p.parse_args().dataset)
