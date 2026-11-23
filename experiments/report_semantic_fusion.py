"""Audit and summarize the semantic fusion follow-up against frozen baselines."""

from collections import defaultdict
import csv
import json
from statistics import mean, stdev

import numpy as np

from evaluation import process_metrics
from semantic_fusion import ROOT, RUN, SEEDS, VIEWS


GRAPH_METHODS = ('magic_adapted', 'protrap_adapted')
METHODS = ('fusion', 'semantic_only', 'count_plus_gru', 'semantic_plus_count')
STRUCTURAL = 'structural_fusion'
NO_STRUCTURE = 'structural_fusion_without_structure'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def baselines():
    groups = defaultdict(lambda: defaultdict(list))
    with (RUN / 'experiment_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['AP_all_mean']:
                groups[(row['dataset'], row['view'])][row['method']].append(float(row['AP_all_mean']))
    with (RUN / 'heldout_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['AP_all']:
                groups[(row['dataset'], row['date'])][row['method']].append(float(row['AP_all']))
    return {scenario: {method: mean(values) for method, values in entries.items()}
            for scenario, entries in groups.items()}


def verify_scores(folder, references):
    per_seed = []
    for dataset in ('theia', 'cadets'):
        source = folder / dataset
        data = load(source / 'results.json')
        if data['status'] != 'COMPLETED':
            raise ValueError(f'{dataset} did not complete')
        calibration = load(source / 'calibration.json')
        cache = ROOT / 'data/cache' / (dataset + '_longoutage_v1')
        for row in data['rows']:
            date, view, seed, method = (row[name] for name in ('date', 'view', 'seed', 'method'))
            is_heldout = view == 'heldout_clean'
            cohort_file = ROOT / 'data/cache/heldout' / dataset / date / 'cohort.json' if is_heldout else cache / 'cohort.json'
            cohort = load(cohort_file)
            archive = source / (date + '_' + view) / f'{seed}_scores.npz'
            with np.load(archive) as values:
                scores = dict(zip(values['actors'].tolist(), values[method].tolist()))
            threshold = calibration['thresholds'][str(seed)][method]
            metrics = process_metrics(cohort['universe'], cohort['positive_processes'], scores, threshold=threshold)
            if metrics != row['metrics']:
                raise ValueError(f'Failed metric replay: {dataset} {date} {view} {seed} {method}')
            reference = references[(dataset, date if is_heldout else view)]
            per_seed.append(dict(dataset=dataset, date=date, view=view, seed=seed, method=method,
                AP_all=metrics['AP_all'], recall_at_50=metrics['top_50']['recall'],
                precision_at_50=metrics['top_50']['precision'], TP_at_50=metrics['top_50']['tp'],
                FP_at_50=metrics['top_50']['fp'], positives=metrics['positive_processes'],
                positive_scoreable=metrics['positive_scoreable'], threshold_FP=metrics['frozen_threshold']['fp'],
                threshold_TP=metrics['frozen_threshold']['tp'], threshold=threshold,
                GRU_block_AP=reference.get('GRU_block'), isolation_forest_AP=reference.get('isolation_forest'),
                magic_adapted_AP=reference.get('magic_adapted'), protrap_adapted_AP=reference.get('protrap_adapted')))
    return per_seed


def aggregate(rows, references, methods):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['date'], row['view'], row['method'])].append(row)
    if len(groups) != len(methods) * (2 * len(VIEWS) + 3):
        raise ValueError('Incomplete follow-up grid')
    summary = []
    for (dataset, date, view, method), entries in groups.items():
        if sorted(row['seed'] for row in entries) != list(SEEDS):
            raise ValueError('Unexpected seed grid')
        reference = references[(dataset, date if view == 'heldout_clean' else view)]
        best_method = max(reference, key=reference.get)
        item = dict(dataset=dataset, date=date, view=view, method=method, n=len(entries),
                    positives=entries[0]['positives'], positive_scoreable=entries[0]['positive_scoreable'])
        for field in ('AP_all', 'recall_at_50', 'precision_at_50', 'TP_at_50', 'FP_at_50', 'threshold_FP', 'threshold_TP'):
            values = [row[field] for row in entries]
            item[field + '_mean'] = mean(values)
            item[field + '_sd'] = stdev(values)
        item.update({name: entries[0][name] for name in ('GRU_block_AP', 'isolation_forest_AP', 'magic_adapted_AP', 'protrap_adapted_AP')})
        item.update(best_baseline=best_method, best_baseline_AP=reference[best_method],
                    delta_vs_best=item['AP_all_mean'] - reference[best_method])
        if all(name in reference for name in GRAPH_METHODS):
            best_graph = max(reference[name] for name in GRAPH_METHODS)
            item.update(delta_vs_best_graph=item['AP_all_mean'] - best_graph,
                        seeds_beating_both_graphs=sum(row['AP_all'] > best_graph for row in entries))
        else:
            item.update(delta_vs_best_graph=None, seeds_beating_both_graphs=None)
        summary.append(item)
    return summary


def fmt(value):
    return '—' if value is None else f'{value:.5f}'


def report(summary, output, structure):
    fusion = [row for row in summary if row['method'] == 'fusion']
    shared = [row for row in fusion if row['delta_vs_best_graph'] is not None]
    graph_wins = sum(row['delta_vs_best_graph'] > 0 for row in shared)
    graph_wins_all_seeds = sum(row['seeds_beating_both_graphs'] == len(SEEDS) for row in shared)
    all_wins = sum(row['delta_vs_best'] > 0 for row in fusion)
    rows = ['# 路径语义与时序证据融合实验', '',
            f'融合方法以三种子平均 AP 在 {graph_wins}/{len(shared)} 个可比较条件超过两种本机图基线，'
            f'其中 {graph_wins_all_seeds} 个条件的三个种子均超过两种图基线。'
            f'纳入频次、Markov、Isolation Forest 和既有 GRU 变体后，在 {all_wins}/{len(fusion)} 个条件超过此前最高 AP。', '',
            '## 方法与比较范围', '',
            '保留三种子时间块 GRU 和计数 Isolation Forest。语义分支从良性训练日统计进程路径、文件访问和父子进程路径组合的支持频次。'
            '每个进程以最罕见的文件交互和父子路径组合形成分数。路径按完整路径、文件名、目录分层统计。'
            '各证据用良性验证分数的尾部位置校准，再取语义与计数中的较大值，加上较小值的 0.25 和 GRU 证据的 0.125。'
            '每个种子的阈值为良性验证融合分数的第 99 百分位。语义参数和系数在本轮评分前设定，已有测试窗口和结果此前已用于诊断，所以本轮属于探索性复测。', '',
            '模型使用数据库导出的静态路径字段，只为当前窗口可见事件所涉及的进程和文件取值。'
            '相同目标的逐事件历史缺失条件仅改变 GRU 历史，路径及计数从共同的窗口目标得到。'
            '全流缺失严格使用先删失后保留的目标。评价是窗口结束后的进程排序，静态路径的逐事件可得性尚未验证。'
            '相对现有图基线，输入特征有所增加。图基线沿用已完成的本机适配版。', '',
            '## 与两种图基线共有的条件', '',
            '| 数据集 | 日期 | 条件 | 融合 AP（均值 ± SD） | MAGIC AP | PRO-TRAP AP | Isolation Forest AP | 融合 Recall@50 |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    for row in shared:
        rows.append(f"| {row['dataset'].upper()} | {row['date']} | {row['view']} | "
                    f"{row['AP_all_mean']:.5f} ± {row['AP_all_sd']:.5f} | {fmt(row['magic_adapted_AP'])} | "
                    f"{fmt(row['protrap_adapted_AP'])} | {fmt(row['isolation_forest_AP'])} | {row['recall_at_50_mean']:.1%} |")
    rows.extend(['', '## 消融', '',
                 '| 数据集 | 日期 | 条件 | 完整融合 AP | 仅语义 AP | 计数 + GRU AP | 语义 + 计数 AP |',
                 '|---|---|---|---:|---:|---:|---:|'])
    lookup = {(row['dataset'], row['date'], row['view'], row['method']): row['AP_all_mean'] for row in summary}
    for row in fusion:
        prefix = (row['dataset'], row['date'], row['view'])
        cells = ' | '.join(fmt(lookup[(*prefix, method)]) for method in METHODS)
        rows.append(f"| {row['dataset'].upper()} | {row['date']} | {row['view']} | {cells} |")
    rows.extend(['', '三种子只对应 GRU 检查点的训练随机性。计数、语义画像和图基线在同场景中各使用一套固定模型。'
                 'CSV 包含完整的告警命中数、精确率、冻结阈值误报及全部基线对比。'])
    if structure:
        structural_fusion = [row for row in structure if row['method'] == STRUCTURAL]
        ablation = {(row['dataset'], row['date'], row['view']): row['AP_all_mean']
                    for row in structure if row['method'] == NO_STRUCTURE}
        common = [row for row in structural_fusion if row['delta_vs_best_graph'] is not None]
        v2_graph_wins = sum(row['delta_vs_best_graph'] > 0 for row in common)
        v2_all_wins = sum(row['delta_vs_best'] > 0 for row in structural_fusion)
        v2_seed_wins = sum(row['seeds_beating_both_graphs'] == len(SEEDS) for row in common)
        rows.extend(['', '## 可见交互结构补测', '',
            '在第一版后增加基于不同交互对象数量、边类型构成、对象关联度和活动跨度的结构分支。'
            'Isolation Forest 在对应 UTC 时段的良性训练进程上拟合。用良性验证数据校准结构分数后，取语义、计数和结构的最大值，'
            '加上三者中位数的 0.125 及 GRU 分数的 0.0625。所有条件使用原有训练日、可见目标和原始进程分母。'
            f'第二版在 {v2_graph_wins}/{len(common)} 个共有条件同时超过两种图基线，三个种子均超过的条件有 {v2_seed_wins} 个，'
            f'在 {v2_all_wins}/{len(structural_fusion)} 个条件超过此前最高 AP。'
            '这一版同样属于复用已见窗口的探索性补测。去掉结构的对照沿用第二版融合式，将结构证据置零。', '',
            '| 数据集 | 日期 | 条件 | 第一版 AP | 第二版 AP | 去掉结构 AP | 最强已有基线 AP |',
            '|---|---|---|---:|---:|---:|---:|'])
        for row in structural_fusion:
            scenario = (row['dataset'], row['date'], row['view'])
            initial = lookup[(*scenario, 'fusion')]
            rows.append(f"| {row['dataset'].upper()} | {row['date']} | {row['view']} | {fmt(initial)} | "
                        f"{fmt(row['AP_all_mean'])} | {fmt(ablation[scenario])} | {row['best_baseline']} {fmt(row['best_baseline_AP'])} |")
    output.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print(json.dumps(dict(graph_wins=graph_wins, graph_conditions=len(shared), graph_wins_all_seeds=graph_wins_all_seeds,
                         all_baseline_wins=all_wins, conditions=len(fusion))), flush=True)


def main():
    folder = ROOT / 'runs/semantic_fusion_v1'
    structural_folder = ROOT / 'runs/semantic_fusion_v2'
    references = baselines()
    seed_rows = verify_scores(folder, references)
    summary = aggregate(seed_rows, references, METHODS)
    write_csv(folder / 'metrics.csv', seed_rows)
    write_csv(folder / 'summary.csv', summary)
    structure = []
    if all((structural_folder / dataset / 'results.json').is_file() for dataset in ('theia', 'cadets')):
        structure_rows = verify_scores(structural_folder, references)
        structure = aggregate(structure_rows, references, (STRUCTURAL, NO_STRUCTURE))
        write_csv(structural_folder / 'metrics.csv', structure_rows)
        write_csv(structural_folder / 'comparison.csv', summary + structure)
    report(summary, ROOT / 'refine-logs/SEMANTIC_FUSION_RESULTS.md', structure)


if __name__ == '__main__':
    main()
