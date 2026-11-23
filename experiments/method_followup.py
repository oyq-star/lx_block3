"""Matched aggregation and forest ablations with a separate benign threshold audit."""

import argparse
from collections import defaultdict
from contextlib import closing
import csv
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.stats import ks_2samp, spearmanr
from sklearn.base import clone

from add_stress_views import utc_ns
from causal_data import EventDay
from count_baseline import features as count_features
from evaluation import process_metrics
from semantic_fusion import (
    ROOT, RUN, SEEDS, VIEWS, WINDOWS, fit_semantics, load, path_index, save_json,
    semantic_evidence, semantic_tail, tail,
)
from structural_fusion import features as structure_features


OUTPUT = ROOT / 'runs/method_followup_v1'
MODES = ('max', 'top5', 'mean')
K = 5
FUSION_RULES = ('maximum','mean','median','weighted_path50','weighted_path25','weighted_path75','max_median')


@dataclass
class Case:
    date: str
    label: str
    group: str
    folder: Path
    nll_paths: tuple
    cohort: dict = None


def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reduction(values, begins):
    ends = np.r_[begins[1:],len(values)]
    avg = np.add.reduceat(values,begins,dtype=np.float64)/(ends-begins)
    top = np.empty(len(begins))
    for i,(left,right) in enumerate(zip(begins,ends)):
        portion = values[left:right]
        count = min(K,len(portion))
        top[i] = np.partition(portion,len(portion)-count)[-count:].mean(dtype=np.float64)
    return dict(max=np.maximum.reduceat(values,begins),top5=top,mean=avg)


def by_actor(edge_actors, values, actors):
    result = {mode:np.full(len(actors),np.nan) for mode in MODES}
    indices = np.searchsorted(actors,edge_actors)
    safe = np.minimum(indices,len(actors)-1)
    present = (indices<len(actors)) & (actors[safe]==edge_actors)
    ids,v = indices[present],values[present]
    if len(ids):
        order = np.argsort(ids,kind='stable')
        sorted_ids = ids[order]
        first = np.r_[0,np.flatnonzero(sorted_ids[1:]!=sorted_ids[:-1])+1]
        for mode,score in reduction(v[order],first).items():
            result[mode][sorted_ids[first]]=score
    return result


def path_aggregates(source, case, indices, semantic_args, actors):
    ids,image,files,creation = semantic_evidence(source,case.date,indices,*semantic_args)
    np.testing.assert_array_equal(ids,actors)
    per_file,per_creation=(by_actor(*group,actors) for group in (files,creation))
    return {mode:np.column_stack((image,per_file[mode],per_creation[mode])) for mode in MODES}


def join_channels(matrix, mode):
    present = np.isfinite(matrix)
    if mode=='max':
        joined = np.max(np.where(present,matrix,-np.inf),axis=1)
    else:
        joined = np.nansum(matrix,axis=1)/np.maximum(present.sum(axis=1),1)
    return np.where(present.any(axis=1),joined,np.nan)


def path_scores(reference, values):
    result = {mode:semantic_tail(reference[mode],values[mode]) for mode in MODES}
    for mode in ('max','mean'):
        result[f'raw_{mode}_then_tail'] = tail(join_channels(reference['max'],mode),join_channels(values['max'],mode))
    return result


def design_matrix(day, indices, mapping, vocabulary):
    actors = day.events['actor'][indices]
    ids, count = count_features(actors,mapping[day.raw_tokens(indices)],vocabulary)
    struct_ids, structure = structure_features(day,indices,mapping,vocabulary)
    np.testing.assert_array_equal(ids,struct_ids)
    return ids, {'Count':count,'Structure':structure}


def feature_groups(vocab):
    return {
        'Count':dict(token_count=slice(0,vocab),token_proportion=slice(vocab,2*vocab),
                     total_events=slice(2*vocab,2*vocab+1)),
        'Structure':dict(distinct_edges=[*range(vocab),2*vocab],edge_event_ratio=slice(vocab,2*vocab),
                         total_events=[2*vocab+1],active_span=[2*vocab+2],
                         object_degree=[2*vocab+3],degree_one=[2*vocab+4]),
    }


def forest_inputs(source, split, mapping, vocabulary):
    windows = []
    for day in split['train_dates']:
        begin,end = (utc_ns(day+split[key][10:]) for key in ('test_start_utc','test_end_utc'))
        with closing(EventDay(source,day)) as logs:
            idx = logs.target_range(begin,end)
            if len(idx):
                _, matrices = design_matrix(logs,idx,mapping,vocabulary)
                windows.append(matrices)
    return {name:np.vstack([entry[name] for entry in windows]) for name in ('Count','Structure')}


def ablated_models(forests, inputs, groups):
    fitted = {}
    for branch, definitions in groups.items():
        matrix = inputs[branch]
        for label, columns in definitions.items():
            exclude = np.zeros(matrix.shape[1],dtype=bool)
            exclude[columns]=True
            keep=~exclude
            fitted[branch,label] = (keep,clone(forests[branch]).fit(matrix[:,keep]))
    return fitted


def evidence(case, source, mapping, vocab, semantic_args, static_cache, forests, ablated):
    indices = np.load(case.folder/'indices.npy',mmap_mode='r')
    actor_rows = np.load(case.folder/'actors.npy',mmap_mode='r')
    begins = np.r_[0,np.flatnonzero(actor_rows[1:]!=actor_rows[:-1])+1]
    actors = actor_rows[begins]
    key = (case.date,'primary_fixed_targets') if case.label in {'clean','iid','dispersed','block','unseen300'} else (case.date,case.label)
    if key not in static_cache:
        with closing(EventDay(source,case.date)) as day:
            ids, matrices = design_matrix(day,indices,mapping,vocab)
        np.testing.assert_array_equal(ids,actors)
        paths=path_aggregates(source,case,indices,semantic_args,actors)
        forest_scores={name:-model.score_samples(matrices[name]) for name,model in forests.items()}
        dropped={(name,group):-model.score_samples(matrices[name][:,cols])
                 for (name,group),(cols,model) in ablated.items()}
        static_cache[key]=(np.array(actors),matrices,paths,forest_scores,dropped)
    ids, matrices, paths, forest_scores, dropped = static_cache[key]
    np.testing.assert_array_equal(ids,actors)
    sequences={}
    for seed,path in zip(SEEDS,case.nll_paths):
        vals=np.load(path,mmap_mode='r')
        if len(vals)!=len(indices) or not np.isfinite(vals).all():
            raise ValueError(f'NLL cache does not align with targets: {case.label}, seed {seed}')
        sequences[seed]=reduction(vals,begins)
    sizes=np.diff(np.r_[begins,len(indices)])
    return dict(actors=ids,events=sizes,features=matrices,path=paths,forest=forest_scores,
                dropped=dropped,sequence=sequences)


def scenarios(dataset, split, cache):
    primary=RUN/dataset
    date=split['test_date']
    cohort=load(cache/'cohort.json')
    for view in VIEWS:
        yield Case(date,view,'original',cache/f'test_{view}',
                   tuple(primary/f'{seed}_block/{view}_event_nll.npy' for seed in SEEDS),cohort)
    for source,date,_,_ in WINDOWS:
        if source!=dataset:
            continue
        folder=ROOT/'data/cache/heldout'/source/date
        cohort=load(folder/'cohort.json')
        yield Case(date,'heldout_clean','original',folder,
                   tuple(RUN/'heldout'/source/date/f'{seed}_block_event_nll.npy' for seed in SEEDS),cohort)
        before=ROOT/'runs/outage_additional_v1'/source/date
        for name in sorted({r['case'] for r in load(before/'results.json')['rows']}):
            folder=before/name
            yield Case(date,name,'additional_outage',folder,
                       tuple(folder/f'{seed}_nll.npy' for seed in SEEDS),cohort)


def fuse(paths, count, structure):
    matrix=np.stack((paths,count,structure))
    maximum=matrix.max(axis=0)
    median=np.median(matrix,axis=0)
    return {
        'maximum':maximum, 'mean':matrix.mean(axis=0), 'median':median,
        **{f'weighted_path{percent}':(percent/100)*paths+(1-percent/100)*.5*(count+structure)
           for percent in (25,50,75)},
        'max_median':maximum+.125*median,
    }


def calibrated(reference, values):
    paths=path_scores(reference['path'],values['path'])
    forests={m:tail(reference['forest'][m],s) for m,s in values['forest'].items()}
    sequences={seed:{mode:tail(reference['sequence'][seed][mode],s) for mode,s in sample.items()}
               for seed,sample in values['sequence'].items()}
    rules=fuse(paths['max'],forests['Count'],forests['Structure'])
    for seed in SEEDS:
        rules[f'original_full_{seed}']=rules['max_median']+.0625*sequences[seed]['max']
    return paths,forests,sequences,rules


def comparison_rows(dataset, case, values, reference, split_reference, thresholds, expected):
    paths,forests,sequences,rules=calibrated(reference,values)
    ids=values['actors'].tolist()
    labels=case.cohort
    prefix=dict(dataset=dataset,date=case.date,view=case.label,grid=case.group)

    def measure(study, rule, variant, scores, seed=None, limit=None):
        metrics=process_metrics(labels['universe'],labels['positive_processes'],dict(zip(ids,scores.tolist())),
                                threshold=limit)
        if study=='fusion' and rule=='original_full' and variant=='original':
            np.testing.assert_allclose(metrics['AP_all'],expected[case.date,case.label,seed],rtol=0,atol=1e-12)
        operating=metrics.get('frozen_threshold',{})
        return dict(**prefix,study=study,rule=rule,variant=variant,seed=seed,AP=metrics['AP_all'],
                    positives=metrics['positive_processes'],scoreable_positives=metrics['positive_scoreable'],
                    TP_at_50=metrics['top_50']['tp'],recall_at_50=metrics['top_50']['recall'],
                    threshold=limit,alert_TP=operating.get('tp'),alert_FP=operating.get('fp'))

    for mode,vector in paths.items():
        yield measure('path','alone',mode,vector)
        yield measure('path','window_max',mode,np.maximum.reduce((vector,forests['Count'],forests['Structure'])))
    for seed in SEEDS:
        for mode in MODES:
            seq=sequences[seed][mode]
            yield measure('sequence','alone',mode,seq,seed=seed)
            yield measure('sequence','original_fusion',mode,rules['max_median']+.0625*seq,seed=seed)
        yield measure('fusion','original_full','original',rules[f'original_full_{seed}'],seed=seed)
    for rule in FUSION_RULES:
        yield measure('fusion',rule,'original',rules[rule])
    _,_,_,separate=calibrated(split_reference,values)
    for rule,vector in separate.items():
        seed=int(rule.rsplit('_',1)[1]) if rule.startswith('original_full_') else None
        label='original_full' if seed else rule
        yield measure('fusion',label,'separate_benign',vector,seed=seed,limit=thresholds[rule])
    for name,score in values['forest'].items():
        yield measure('forest',name,'full',score)
        yield measure('forest_fusion',name,'full',rules['maximum'])
    for (name,group),dropped in values['dropped'].items():
        other='Structure' if name=='Count' else 'Count'
        calibrated_drop=tail(reference['dropped'][name,group],dropped)
        yield measure('forest',name,group,dropped)
        yield measure('forest_fusion',name,group,np.maximum.reduce((paths['max'],forests[other],calibrated_drop)))


def benign_checks(dataset, benign):
    p0,f0,s0,r0=calibrated(benign['reference'],benign['reference'])
    p1,f1,s1,r1=calibrated(benign['reference'],benign['threshold'])
    p2,f2,s2,r2=calibrated(benign['reference'],benign['audit'])
    rng=np.random.default_rng(17)
    thresholds={name:float(np.quantile(a,.99)) for name,a in r1.items()}
    rows=[]; survival=[]
    for role,scores in [('reference',(p0,f0,s0,r0)),('threshold',(p1,f1,s1,r1)),('audit',(p2,f2,s2,r2))]:
        path,forest,seq,rules=scores
        families={
            **{f'path/{m}':s for m,s in path.items()},
            **{f'branch/{m}':s for m,s in forest.items()},
            **{f'sequence/{seed}/{m}':s for seed,variants in seq.items() for m,s in variants.items()},
            **{f'fusion/{name}':s for name,s in rules.items()},
        }
        for name,x in families.items():
            cutoff=thresholds[name[7:]] if name.startswith('fusion/') else None
            q50,q90,q99,q999=np.quantile(x,[.5,.9,.99,.999])
            p05=p95=None
            if role=='threshold' and cutoff is not None:
                samples=x[rng.integers(0,len(x),size=(200,len(x)))]
                p05,p95=np.quantile(np.quantile(samples,.99,axis=1),[.05,.95])
            rows.append(dict(dataset=dataset,role=role,score=name,n=len(x),p50=q50,p90=q90,p99=q99,
                             p999=q999,max=x.max(),frozen_threshold=cutoff,
                             alarm_count=int(np.sum(x>=cutoff)) if cutoff is not None else None,
                             alarm_rate=float(np.mean(x>=cutoff)) if cutoff is not None else None,
                             resampled_threshold_p05=p05,resampled_threshold_p95=p95))
            for point in np.arange(0,10.25,.25):
                survival.append(dict(dataset=dataset,role=role,score=name,n=len(x),point=point,
                                     exceedance_rate=float(np.mean(x>=point))))
    return thresholds,rows,survival


def shifts(dataset,case,values,train,groups):
    for branch,features in values['features'].items():
        for group,columns in groups[branch].items():
            for index in np.arange(features.shape[1])[columns]:
                ref,obs=train[branch][:,index],features[:,index]
                tq=np.quantile(ref,[.5,.9,.99])
                vq=np.quantile(obs,[.5,.9,.99])
                yield dict(dataset=dataset,date=case.date,branch=branch,group=group,index=index,
                           train_n=len(ref),test_n=len(obs),train_zero=np.mean(ref==0),test_zero=np.mean(obs==0),
                           train_p50=tq[0],train_p90=tq[1],train_p99=tq[2],test_p50=vq[0],test_p90=vq[1],
                           test_p99=vq[2],KS=ks_2samp(ref,obs).statistic)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('theia','cadets'),required=True)
    dataset=parser.parse_args().dataset
    output=OUTPUT/dataset
    cache=ROOT/'data/cache'/f'{dataset}_longoutage_v1'
    source=ROOT/'data/processed'/f'{dataset}_e3'
    v1=ROOT/'runs/semantic_fusion_v1'/dataset
    v2=ROOT/'runs/semantic_fusion_v2'/dataset
    split=load(cache/'split.json')
    mapping=np.load(cache/'token_map.npy')
    vocabulary=load(cache/'dataset.json')['vocabulary_size']
    schema=load(source/'schema.json')
    roles=[int(i) for i,n in schema['relation_names'].items() if n in {'EVENT_CLONE','EVENT_FORK'}]
    subjects,files=path_index(dataset,source,v1)
    support,documents=fit_semantics(source,split['train_dates'],subjects,files,roles)
    semantic_args=subjects,files,roles,support,documents
    forests=dict(Count=joblib.load(RUN/dataset/'isolation_forest/model.joblib'),
                 Structure=joblib.load(v2/'structure_forest.joblib'))
    train=forest_inputs(source,split,mapping,vocabulary)
    groups=feature_groups(vocabulary)
    ablated=ablated_models(forests,train,groups)
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'forest_parameters.json',dict(library='scikit-learn',version=sklearn.__version__,
              parameters={k:v.get_params() for k,v in forests.items()},
              train_actors={k:len(v) for k,v in train.items()},standardization='none; count logs and ratios as specified'))
    static={}

    def calculate(case):
        return evidence(case,source,mapping,vocabulary,semantic_args,static,forests,ablated)

    original_window=Case(split['validation_date'],'original_reference','benign',cache/'validation',
        tuple(RUN/dataset/f'{seed}_block/validation_nll.npy' for seed in SEEDS))
    original=calculate(original_window)
    benign={}
    for role in ('reference','threshold','audit'):
        folder=output/'benign'/role
        protocol=load(folder/'scores_complete.json')
        benign[role]=calculate(Case(protocol['date'],role,'benign',folder,
                              tuple(folder/f'{seed}_nll.npy' for seed in SEEDS)))
    limits,benign_rows,tail_rows=benign_checks(dataset,benign)
    write_csv(output/'benign_summary.csv',benign_rows)
    write_csv(output/'benign_tails.csv',tail_rows)
    expected={(r['date'],r['view'],r['seed']):r['metrics']['AP_all'] for r in load(v2/'results.json')['rows']
              if r['method']=='structural_fusion'}
    additional=ROOT/'runs/outage_additional_v1'/dataset
    for file in additional.glob('*/results.json'):
        expected.update({(r['date'],r['case'],r['seed']):r['metrics']['AP_all'] for r in load(file)['rows']
                         if r['method']=='Full'})
    rows=[]; distributions=[]; correlations=[]
    for case in scenarios(dataset,split,cache):
        values=calculate(case)
        rows.extend(comparison_rows(dataset,case,values,original,benign['reference'],limits,expected))
        if case.label in ('clean','heldout_clean'):
            distributions.extend(shifts(dataset,case,values,train,groups))
        unannotated=~np.isin(values['actors'],case.cohort['positive_processes'])
        volume=np.log1p(values['events'][unannotated])
        for seed,variants in values['sequence'].items():
            for mode,nll in variants.items():
                rho=spearmanr(volume,nll[unannotated]).correlation
                correlations.append(dict(dataset=dataset,date=case.date,view=case.label,grid=case.group,
                                         seed=seed,aggregation=mode,n=len(volume),rho=rho))
        print(dataset,case.date,case.label,len(values['actors']),flush=True)
    write_csv(output/'per_condition.csv',rows)
    write_csv(output/'feature_shift.csv',distributions)
    write_csv(output/'nll_volume.csv',correlations)
    save_json(output/'status.json',dict(status='COMPLETED',dataset=dataset,rows=len(rows),grids=sorted({x['grid'] for x in rows}),
              conditions=len({(x['date'],x['view']) for x in rows}),spread='sample SD across seeds; fixed-view comparisons',
              thresholds=limits))


if __name__=='__main__':
    main()
