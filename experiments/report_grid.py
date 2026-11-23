"""Audit recorded scores, then publish the two-domain grid as CSV and Markdown."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from evaluation import process_metrics
from heldout_campaigns import stats_scores
from local_grid import MODES, VIEWS


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'idea-stage/evidence'))
from artifact_io import write_artifact

DATASETS=('theia','cadets')
SEEDS=(17,23,42)
VIEWS=VIEWS.split(',')
MATCHED=('iid','dispersed','block')
GRAPHS=('magic_adapted','protrap_adapted')
GRAPH_VIEWS=('clean','global_relation','global_time')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_out(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def cache_audit(cache):
    info={view:load(cache/('test_'+view)/'metadata.json') for view in VIEWS}
    clean=np.load(cache/'test_clean/indices.npy',mmap_mode='r')
    for view in VIEWS:
        folder=cache/('test_'+view)
        assert (folder/'CACHE_COMPLETE').exists(),folder
        ids=np.load(folder/'indices.npy',mmap_mode='r')
        if view.startswith('global_'):
            indices=np.searchsorted(clean,ids)
            assert not len(ids) or (np.max(indices)<len(clean) and np.array_equal(clean[indices],ids)),view
        else:
            assert np.array_equal(clean,ids),view
    counts=[tuple(info[view]['corruption_summary'][key] for key in ('K_raw','K_eff','empty')) for view in MATCHED]
    assert len(set(counts))==1,counts
    return dict(cohort=load(cache/'cohort.json'),views=info,split=load(cache/'split.json'),
                dataset_config=load(cache/'dataset.json'))


def verify(metrics,cohort,scores):
    expected=process_metrics(cohort['universe'],cohort['positive_processes'],scores,
                             threshold=metrics['frozen_threshold']['threshold'])
    assert expected==metrics,(expected,metrics)


def make_row(dataset,method,seed,view,metrics,log=None,inference=None):
    budget=metrics['top_50'];threshold=metrics['frozen_threshold']
    tp,fp,fn=threshold['tp'],threshold['fp'],threshold['fn']
    log=log or {};inference=inference or {};resources=log.get('resources',{})
    p_latency=inference.get('batch_latency_ms_p50_p95',[])
    return dict(dataset=dataset,method=method,seed=seed,view=view,
                protocol='stream' if view.startswith('global_') else 'fixed_targets',
                AP_all=metrics['AP_all'],AP_observed=metrics['AP_observed'],
                coverage=metrics['positive_coverage'],positives=metrics['positive_processes'],
                positive_scoreable=metrics['positive_scoreable'],scoreable=metrics['scoreable_processes'],
                recall_at_50=budget['recall'],precision_at_50=budget['precision'],
                tp_at_50=budget['tp'],fp_at_50=budget['fp'],ties_at_50=budget['boundary_tied_processes'],
                threshold=threshold['threshold'],threshold_tp=tp,threshold_fp=fp,threshold_fn=fn,
                threshold_recall=threshold['recall'],threshold_precision=threshold['precision'],
                threshold_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                training_seconds=log.get('training_seconds'),wall_seconds=log.get('total_wall_seconds',log.get('total_seconds')),
                process_RSS_peak_mib=resources.get('process_RSS_peak_MiB'),
                process_RSS_end_mib=log.get('rss_end_mib'),
                torch_cuda_peak_allocated_mib=resources.get('torch_cuda_allocated_peak_MiB',log.get('peak_allocated_mib')),
                scoring_seconds=inference.get('seconds'),cached_events_per_sec=inference.get('events_per_second'),
                legacy_coarse_batch_p50_ms=p_latency[0] if p_latency else None,
                legacy_coarse_batch_p95_ms=p_latency[1] if p_latency else None)


def gather(run):
    rows=[];missing=[];details={};nll=[]
    for dataset in DATASETS:
        cache=ROOT/'data/cache'/(dataset+'_longoutage_v1')
        audit=cache_audit(cache);cohort=audit['cohort'];details[dataset]=audit
        folder=run/dataset
        stats=folder/'statistics/results.json'
        if not stats.exists():missing.append(f'{dataset}/statistics')
        else:
            with np.load(stats.parent/'fitted_probabilities.npz') as probs:
                for method,result in load(stats)['results'].items():
                    assert set(result['views'])==set(VIEWS),method
                    for name,metrics in result['views'].items():
                        verify(metrics,cohort,stats_scores(cache/('test_'+name),probs,method))
                        rows.append(make_row(dataset,method,17,name,metrics))
        count_path=folder/'isolation_forest/results.json'
        if not count_path.exists():missing.append(f'{dataset}/isolation_forest')
        else:
            count=load(count_path)
            assert count['status']=='COMPLETED' and set(count['views'])==set(VIEWS),count_path
            for name,metrics in count['views'].items():
                scores={int(key):value for key,value in load(count_path.parent/(name+'_process_scores.json')).items()}
                verify(metrics,cohort,scores)
                rows.append(make_row(dataset,'isolation_forest',17,name,metrics,count))
        for method in GRAPHS:
            path=folder/method/'results.json'
            if not path.exists():missing.append(f'{dataset}/{method}');continue
            data=load(path)
            assert data['status']=='COMPLETED' and set(data['views'])==set(GRAPH_VIEWS),path
            for name,result in data['views'].items():
                scores={int(k):v for k,v in load(path.parent/('test_'+name+'_process_scores.json')).items()}
                verify(result['metrics'],cohort,scores)
                assert result['events']==audit['views'][name]['rows']
                rows.append(make_row(dataset,method,17,name,result['metrics'],data,result))
        actor_ids=np.load(cache/'test_clean/actors.npy',mmap_mode='r')
        label=np.isin(actor_ids,cohort['positive_processes'])
        for seed in SEEDS:
            for mode in MODES:
                model=folder/f'{seed}_{mode}'
                path=model/'results.json'
                if not path.exists():missing.append(f'{dataset}/{model.name}');continue
                data=load(path);config=load(model/'config.json')
                assert set(data['views'])==set(VIEWS) and data['status']=='COMPLETED' and data['seed']==seed and data['mode']==mode,path
                assert data['cohort']==cohort and len(data['history'])==3 and config['batch_size']==64,path
                assert all(item['targets']==audit['dataset_config']['train_targets'] for item in data['history']),path
                for name,view in data['views'].items():
                    with (model/(name+'_process_predictions.csv')).open(encoding='utf-8',newline='') as stream:
                        scores={int(entry['process_id']):float(entry['score']) for entry in csv.DictReader(stream) if entry['score']}
                    verify(view['metrics'],cohort,scores)
                    assert view['inference']['rows']==audit['views'][name]['rows']
                    rows.append(make_row(dataset,'GRU_'+mode,seed,name,view['metrics'],data,view['inference']))
                if mode=='clean':
                    original=np.load(model/'clean_event_nll.npy',mmap_mode='r')
                    for name in (*MATCHED,'unseen300'):
                        loss=np.load(model/(name+'_event_nll.npy'),mmap_mode='r')
                        delta=np.subtract(loss,original,dtype=np.float64)
                        for group,subset in [('all',delta),('annotated',delta[label]),('unannotated',delta[~label])]:
                            nll.append(dict(dataset=dataset,seed=seed,view=name,group=group,n=len(subset),
                                mean_NLL_change=float(np.mean(subset)),median_NLL_change=float(np.median(subset)),
                                fraction_increased=float(np.mean(subset>1e-6))))
    return rows,nll,missing,details


def aggregate(rows):
    groups={}
    for item in rows:groups.setdefault((item['dataset'],item['method'],item['view']),[]).append(item)
    result=[]
    for (dataset,method,view),items in groups.items():
        agg=dict(dataset=dataset,method=method,view=view,n=len(items))
        for key in ('AP_all','AP_observed','recall_at_50','precision_at_50','coverage','threshold_fp','threshold_fn'):
            values=[item[key] for item in items if item[key] is not None]
            agg[key+'_mean']=float(np.mean(values)) if values else None
            agg[key+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        result.append(agg)
    return result


def markdown(rows,summary,missing,details):
    status='规定矩阵已完成' if not missing else '部分结果：仍有任务未完成'
    output=['# APT 检测实验结果（THEIA_E3 / CADETS_E3）','',
            f'**{status}**。{len(rows)} 条数据行（数据集×方法×种子×评估条件）。','',
            '按用户要求补跑此前因 M1 暂缓的实验。首轮结果独立保留。GRU 用相同良性样本，训练 3 轮、批量 64、同种子配对，按良性验证损失选模型；三个种子衡量训练随机性。Isolation Forest 用训练日对应时段的全量进程计数，属于使用不同聚合特征的补充统计基线。','']
    for dataset,entry in details.items():
        cohort=entry['cohort'];views=entry['views'];split=entry['split']
        output.extend([f"- {dataset.upper()}：{split['test_start_utc']} 至 {split['test_end_utc']}，{cohort['original_target_events']:,} 个目标、{len(cohort['universe']):,} 个进程、{len(cohort['positive_processes'])} 个标注攻击进程。",
            f"- {dataset.upper()} 长时块 / 未见 300 秒块改变 {views['block']['rows']-views['block']['corruption_summary']['empty']:,} / {views['unseen300']['rows']-views['unseen300']['corruption_summary']['empty']:,} 个目标上下文。全流关系停采 / 纯时间停采删除 {views['global_relation']['deleted_targets']:,} / {views['global_time']['deleted_targets']:,} 个目标。"])
    output.extend(['','固定目标：clean、iid、dispersed、block、unseen300。三种长时缺失匹配 K_raw、K_eff，iid 指均匀无放回。全流条件先删再建模，仍使用原始进程及标签分母：global_relation 是 1799 秒训练最高频关系，global_time 是 300 秒全关系。全流两条件强度不同。','',
        'MAGIC 使用 PIDSMaker 公开编码器与训练目标；PRO-TRAP 使用发表方法的公开 edge GAT 和时间视图子图对比。两者均按本机资源适配图分片与阈值，使用单种子，只比较 clean 和全流条件，保留各自分片内回溯上下文。不宣称复现原论文数字。','',
        '## 进程级 AP_all','',
        '| 数据集 | 方法 | n | clean | iid | dispersed | block | unseen300 | global_relation | global_time |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for dataset in DATASETS:
        for method in ['frequency','markov','isolation_forest']+['GRU_'+mode for mode in MODES]+list(GRAPHS):
            found={r['view']:r for r in summary if r['dataset']==dataset and r['method']==method}
            if not found:continue
            cells=[f"{found[view]['AP_all_mean']:.5f}" if view in found else '—' for view in VIEWS]
            output.append(f"| {dataset} | {method} | {max(item['n'] for item in found.values())} | "+' | '.join(cells)+' |')
    output.extend(['','## 三种子配对 ΔAP：块训练减去控制','',
        '| 数据集 | 控制 | 块缺失测试上的逐种子差值（17 / 23 / 42） | 正向数 |',
        '|---|---|---|---:|'])
    lookup={(r['dataset'],r['method'],r['seed'],r['view']):r for r in rows}
    robust=True
    for dataset in DATASETS:
        for control in ('clean','iid','dispersed','crop'):
            delta=[]
            for seed in SEEDS:
                after=lookup.get((dataset,'GRU_block',seed,'block'))
                before=lookup.get((dataset,'GRU_'+control,seed,'block'))
                if before and after:delta.append(after['AP_all']-before['AP_all'])
            if control!='clean' and (len(delta)!=3 or not all(value>0 for value in delta)):robust=False
            output.append(f"| {dataset} | {control} | {', '.join(f'{item:+.5f}' for item in delta)} | {sum(item>0 for item in delta)}/{len(delta)} |")
    if missing:output+=['','未完成：'+', '.join(missing)+'。']
    elif not robust:output+=['','块训练未在两套数据上稳定超过均匀、同关系分散和时间裁剪对照，不能宣称时间块训练有独立优势。']
    else:output+=['','块训练相对三种对照在两套数据上均呈正向；仍需检查 clean 退化及单场景限制。']
    if not missing:
        output+=['','无缺失条件下，块训练相对 clean GRU 的配对变化：','']
        for dataset in DATASETS:
            delta_ap=[];delta_recall=[]
            for seed in SEEDS:
                before=lookup[(dataset,'GRU_clean',seed,'clean')]
                after=lookup[(dataset,'GRU_block',seed,'clean')]
                delta_ap.append(after['AP_all']-before['AP_all'])
                delta_recall.append(after['recall_at_50']-before['recall_at_50'])
            output.append(f'- {dataset.upper()} 平均 ΔAP {np.mean(delta_ap):+.5f}；平均 ΔRecall@50 {np.mean(delta_recall):+.2%}。'
                          f'逐种子超过预设退化线（AP −0.01 或 Recall@50 −0.05）：'
                          f'{sum(a < -.01 or r < -.05 for a,r in zip(delta_ap,delta_recall))}/3。')
    output+=['','各 seed、Recall@50、Precision@50、冻结阈值、混淆计数、资源及批延迟见全量 CSV；SD 是样本标准差。不能把缓存推断视为端到端在线延迟。CADETS 首窗口只有两个标注攻击进程。']
    return '\n'.join(output)+'\n'


def heldout_section(path):
    if not path.is_file():return '',0
    with path.open(encoding='utf-8-sig',newline='') as stream:
        records=list(csv.DictReader(stream))
    assert len(records)==60,len(records)
    output=['','## 额外三场攻击窗口：冻结模型，无缺失评估','',
        '窗口覆盖配置中剩余三场有进程标注的攻击。阈值、词表、参考样本、模型均复用主实验，不在这些窗口更新；没有把缺标注日视为明确良性。以下 AP 对 GRU 求种子平均，其他方法为单次。','',
        '| 数据集、日期 | 标注进程 | Markov | Isolation Forest | Clean GRU | Block GRU | MAGIC 适配 | PRO-TRAP 适配 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for dataset,date in sorted({(row['dataset'],row['date']) for row in records}):
        group=[row for row in records if row['dataset']==dataset and row['date']==date]
        assert len(group)==20,(dataset,date)
        means=[]
        for method in ['markov','isolation_forest','GRU_clean','GRU_block',*GRAPHS]:
            values=[float(row['AP_all']) for row in group if row['method']==method and row['AP_all']]
            means.append(f'{np.mean(values):.5f}' if values else '—')
        output.append(f"| {dataset}, {date} | {group[0]['positives']} | "+' | '.join(means)+' |')
    output+=['','这些是预置两小时切片，并不等于覆盖所有 DARPA E3 整日或其他数据集；完整缺失矩阵只在主表两窗口测量。','']
    return '\n'.join(output),len(records)


def diagnostic_section(rows,nll,details,run):
    out=['','## 配对损失和资源记录','',
         '以下是 clean GRU 同目标的平均 NLL 增量（缺失 − clean），先逐种子计算，再取种子均值。它衡量预测偏移，不能替代检测 AP。','',
         '| 数据集 | iid | 同关系分散 | 关系时间块 | 未见 300 秒 |',
         '|---|---:|---:|---:|---:|']
    for dataset in DATASETS:
        means=[]
        for view in (*MATCHED,'unseen300'):
            values=[item['mean_NLL_change'] for item in nll
                    if item['dataset']==dataset and item['view']==view and item['group']=='all']
            means.append(f'{np.mean(values):+.5f}' if values else '—')
        out.append(f'| {dataset} | '+' | '.join(means)+' |')
    out.extend(['','| 数据集 | 训练墙钟时间范围（s） | PyTorch 峰值 allocated（MiB） | 最大采样 RSS（MiB） |',
                '|---|---:|---:|---:|'])
    for dataset in DATASETS:
        clean=[r for r in rows if r['dataset']==dataset and r['method'].startswith('GRU_') and r['view']=='clean']
        if not clean:continue
        times=[r['training_seconds'] for r in clean]
        gpu=max(r['torch_cuda_peak_allocated_mib'] for r in clean)
        rss=max(r['process_RSS_peak_mib'] for r in clean)
        out.append(f'| {dataset} | {min(times):.1f}–{max(times):.1f} | {gpu:.1f} | {rss:.1f} |')
    profile_path=run/'inference_profile.json'
    if profile_path.exists():
        out.extend(['','主队列的 Windows 计时器以约 15.6 ms 为单位，只保留在 CSV 的 legacy_coarse 字段。用 QueryPerformanceCounter 对 seed 17 冻结模型重测 1024 条一批的缓存推断（包含计分聚合）。','',
                    '| 数据集 | 训练变体 | 视图 | P50/P95（ms） | 事件/秒 |','|---|---|---|---:|---:|'])
        for sample in load(profile_path)['rows']:
            p50,p95=sample['batch_latency_ms_p50_p95']
            out.append(f"| {sample['dataset']} | {sample['mode']} | {sample['view']} | {p50:.2f} / {p95:.2f} | {sample['events_per_second']:,.0f} |")
    state=load(run/'grid_state.json')
    gpu_time=sum(item.get('wall_seconds',0) for name,item in state['jobs'].items() if not name.endswith('/statistics'))
    out.extend(['',f'队列已记录 GPU 任务墙钟合计 {gpu_time/60:.2f} 分钟（含训练、验证、推断和 CUDA 小样本检查；不是实际 CUDA 忙时）。PRO-TRAP 和统计基线使用 CPU。',
        '进程级 GPU 专用显存无法通过本机 WDDM 的 nvidia-smi 查询获得。表中是 PyTorch allocator 指标；批量推断指标使用预建上下文。两者都不是面向实时部署的资源承诺。','',
        '全流诊断的原始/可见事件数逐进程保存于两个 global 缓存目录的 process_event_counts.npz；上下文回填数、空上下文和删失量在缓存元数据中。逐目标的裁剪基线与两个匹配删失控制职责不同：裁剪使用相同的参考激活/持续时间，但不约束实际 K_eff 等于块。','',
        '重放入口是 experiments/local_grid.py、experiments/count_baseline.py、experiments/protrap_adapted.py 和 experiments/heldout_campaigns.py；主队列的完整命令和退出码在 grid_state.json。事件 NLL、进程得分、阈值、标签和训练曲线保留在各模型目录中。',''])
    return '\n'.join(out)


def publish_results(report,run,supplement_count):
    write_artifact('refine-logs/FULL_EXPERIMENT_RESULTS.md',report,'Full two-domain experiment grid',
                   skill='experiment-bridge',stage='execution')
    track='''# Experiment Tracker — two-domain follow-up

| Run | Result | Evidence |
|---|---|---|
| R001 | Completed | Verified THEIA/CADETS dumps, UUID mapping, schema and real-data CUDA sanity |
| R002 | Completed | Frequency, Markov, Isolation Forest, and clean GRU across two cohorts plus matched deletion diagnostic |
| R003 | Completed | Clean/iid/dispersed/block fixed-budget comparisons, all three seeds |
| R004 | Completed | Temporal crop control; inspired by PRO-TRAP, separate from graph baseline |
| R005 | Completed | CADETS evaluation on fixed cohort |
| R006 | Completed | 30 GRU training jobs: two datasets × five variants × three seeds |
| R007 | Completed (adapted) | MAGIC and PRO-TRAP graph baselines on both datasets with the same clean and stream-censored cohorts. Other published systems are not reproduced |
| R008 | Completed for declared stress views | Unseen 300s relation block, full visible stream relation and all-relation time masks, coverage, refill and cached resources. No collector field or end-to-end online delay claim |
| Supplement | Completed | Frozen checkpoints on three additional annotated campaign windows, clean only |
| B002 | Prior CPU diagnostic | Separate C02 sibling coverage; no C02 detection training |
| B006 | Backup only | C06 not part of the C03 comparison |

Full numbers: FULL_EXPERIMENT_RESULTS.md and runs/full_grid_v1/experiment_metrics.csv.
'''
    write_artifact('refine-logs/EXPERIMENT_TRACKER.md',track,'Completed two-domain experiments and adaptation boundaries',
                   skill='experiment-bridge',stage='execution')
    state_path=ROOT/'refine-logs/BRIDGE_STATE.json'
    state=load(state_path)
    state.update(stage='TWO_DOMAIN_GRID_COMPLETED',report='refine-logs/FULL_EXPERIMENT_RESULTS.md',
                 run=str(run.relative_to(ROOT)),n_gru_grid_jobs=30,n_trained_graph_baselines=4,
                 n_heldout_clean_rows=supplement_count,paper_ready=False)
    write_artifact('refine-logs/BRIDGE_STATE.json',json.dumps(state,indent=2),'Completed follow-up run state',
                   skill='experiment-bridge',stage='execution')
    contract_file='idea-stage/docs/research_contract.md'
    contract=(ROOT/contract_file).read_text(encoding='utf-8').split('\n## 两域完整对照更新')[0]
    update='''## 两域完整对照更新

按用户要求保留原先试验并完成补跑。C03 两域、五 GRU 变体及三种子共 30 次训练，补做频次、Markov、Isolation Forest 和 MAGIC/PRO-TRAP 适配基线，并对后续三场标注窗口用冻结模型复核。主结果和评价边界见 refine-logs/FULL_EXPERIMENT_RESULTS.md。

当前数据不支持将“时间块增强优于同关系分散控制”作为主要论文结论。RTX 3060 上有限成本可训练的结论得到支持，但没有在线处理性能结论。图模型是适配结果，保留原始单窗口试验和原始研究计划，不改写先验预测。
'''
    write_artifact(contract_file,contract.rstrip()+'\n\n'+update,'Research contract evidence update after the full grid',
                   skill='experiment-bridge',stage='execution')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT/'runs/full_grid_v1')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    rows,nll,missing,details=gather(args.run)
    summary=aggregate(rows)
    csv_out(args.run/'experiment_metrics.csv',rows)
    csv_out(args.run/'experiment_summary.csv',summary)
    csv_out(args.run/'paired_nll_summary.csv',nll)
    supplement,supplement_count=heldout_section(args.run/'heldout_metrics.csv')
    payload=dict(status='PARTIAL' if missing else 'COMPLETED',missing=missing,metric_rows=len(rows),
                 heldout_clean_rows=supplement_count,
                 seeds=list(SEEDS),details=details,
                 audit='shared target order, global targets subset, matching quotas counts, reproduce GRU and graph process metrics')
    (args.run/'experiment_manifest.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    report=markdown(rows,summary,missing,details)+diagnostic_section(rows,nll,details,args.run)+supplement
    (args.run/'RESULTS.md').write_text(report,encoding='utf-8')
    if args.publish:
        if missing or not supplement_count:raise RuntimeError('Incomplete results; cannot publish tracker as completed')
        publish_results(report,args.run.resolve(),supplement_count)
    print(json.dumps(dict(rows=len(rows),missing=missing,paired_rows=len(nll))),flush=True)


if __name__=='__main__':main()
