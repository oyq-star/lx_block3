"""Replay paired ablations and fixed fusion rules on the saved process scores."""

from collections import defaultdict
from itertools import product
from statistics import mean, stdev

import numpy as np

from evaluation import process_metrics
from report_semantic_fusion import write_csv
from semantic_fusion import ROOT, SEEDS, load, save_json
from structural_fusion import ABLATION, METHOD, integrate


OUTPUT = ROOT / 'runs/fusion_sensitivity_v1'
BUDGETS = (10, 20, 30, 50, 75, 100)
BRANCHES = ('path', 'count', 'structure', 'gru')
MEDIAN_WEIGHTS = (0., .125, .25)
GRU_WEIGHTS = (0., .0625, .125)


def weighted_window_score(window_scores, sequence, median_weight=.125, gru_weight=.0625):
    return (window_scores.max(axis=0) + median_weight * np.median(window_scores, axis=0)
            + gru_weight * sequence)


def variants(branch_scores):
    window_scores, sequence = branch_scores[:3], branch_scores[3]
    yield 'full', weighted_window_score(window_scores, sequence)
    # Use a zero score for the dropped branch while preserving its position in the rule.
    for index, branch in enumerate(BRANCHES):
        omitted = branch_scores.copy()
        omitted[index] = 0
        yield 'without_' + branch, weighted_window_score(omitted[:3], omitted[3])
    for branch, scores in zip(BRANCHES, branch_scores):
        yield 'only_' + branch, scores
    yield 'equal_mean', branch_scores.mean(axis=0)
    yield 'equal_max', branch_scores.max(axis=0)
    for beta, gamma in product(MEDIAN_WEIGHTS, GRU_WEIGHTS):
        if beta == .125 and gamma in (0., .0625):
            continue
        yield f'weight_{beta:g}_{gamma:g}', weighted_window_score(window_scores, sequence, beta, gamma)


def metric_row(*, dataset, date, view, seed, method, metrics):
    row = dict(dataset=dataset, date=date, view=view, seed=seed, method=method,
               AP=metrics['AP_all'], positives=metrics['positive_processes'],
               observable_positives=metrics['positive_scoreable'])
    for budget in BUDGETS:
        values = metrics[f'top_{budget}']
        for name in ('tp', 'fp', 'recall', 'precision'):
            row[f'{name}@{budget}'] = values[name]
    return row


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in ('dataset', 'date', 'view', 'method'))
        groups[key].append(row)
    summary = []
    for (dataset, date, view, method), entries in groups.items():
        assert sorted(r['seed'] for r in entries) == list(SEEDS)
        row = dict(dataset=dataset, date=date, view=view, method=method, n=len(entries),
                   positives=entries[0]['positives'], observable_positives=entries[0]['observable_positives'])
        for field in ['AP', *(f'{name}@{k}' for k in BUDGETS for name in ('tp', 'fp', 'recall', 'precision'))]:
            values = [r[field] for r in entries]
            row[field + '_mean'] = mean(values)
            row[field + '_sd'] = stdev(values)
        summary.append(row)
    return summary


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in ('theia', 'cadets'):
        v1 = ROOT / 'runs/semantic_fusion_v1' / dataset
        v2 = ROOT / 'runs/semantic_fusion_v2' / dataset
        refs = load(v2 / 'results.json')['rows']
        v2_reference = {(row['date'], row['view'], row['seed'], row['method']): row['metrics'] for row in refs}
        for date, view in sorted({(r['date'], r['view']) for r in refs}):
            cohort_path = (ROOT / 'data/cache/heldout' / dataset / date if view == 'heldout_clean'
                           else ROOT / 'data/cache' / (dataset + '_longoutage_v1'))
            cohort = load(cohort_path / 'cohort.json')
            for seed in SEEDS:
                name = f'{date}_{view}/{seed}_scores.npz'
                with np.load(v1 / name) as first, np.load(v2 / name) as second:
                    actors = first['actors']
                    np.testing.assert_array_equal(actors, second['actors'])
                    assert set(actors).issubset(cohort['universe'])
                    branches = np.stack((first['semantic'], first['count'], second['structure'], first['temporal']))
                    assert np.isfinite(branches).all() and (branches >= 0).all()
                    stored = integrate(*branches)
                    for method in (METHOD, ABLATION):
                        np.testing.assert_allclose(stored[method], second[method], rtol=0, atol=1e-12)
                for method, scores in variants(branches):
                    metrics = process_metrics(cohort['universe'], cohort['positive_processes'],
                                              dict(zip(actors.tolist(), scores.tolist())), budgets=BUDGETS)
                    if method in ('full', 'without_structure'):
                        reference = v2_reference[date, view, seed, METHOD if method == 'full' else ABLATION]
                        for field in ('AP_all', 'positive_scoreable', 'top_10', 'top_50', 'top_100'):
                            np.testing.assert_equal(metrics[field], reference[field])
                    rows.append(metric_row(dataset=dataset, date=date, view=view, seed=seed,
                                           method=method, metrics=metrics))
    summary = summarize(rows)
    methods = sorted({r['method'] for r in rows})
    assert len(summary) == 17 * len(methods) and len(rows) == len(summary) * len(SEEDS)
    write_csv(OUTPUT / 'per_seed.csv', rows)
    write_csv(OUTPUT / 'summary.csv', summary)
    save_json(OUTPUT / 'protocol.json', dict(
        cases=17, checkpoints=SEEDS, methods=methods, budgets=BUDGETS,
        median_weights=MEDIAN_WEIGHTS, gru_weights=GRU_WEIGHTS,
        scope='Exploratory paired inference using earlier fitted checkpoints and calibrated branch scores',
        row_count=len(rows), summary_count=len(summary),
    ))
    print(f'{len(methods)} rules, 17 conditions, {len(rows)} per-seed rows, {len(summary)} summary rows', flush=True)


if __name__ == '__main__':
    main()
