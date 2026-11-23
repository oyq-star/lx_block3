"""Publish the completed THEIA pilot and its plan-based stop decision."""
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'idea-stage/evidence'))
from artifact_io import write_artifact


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def publish(relative, content, description):
    write_artifact(relative, content, description, skill='experiment-bridge', stage='execution')


def main():
    run = 'runs/theia_pilot_v1/'
    sanity = load(run+'neural_stage1/sanity/results.json')
    clean = load(run+'neural_stage1/clean/results.json')
    config = load(run+'neural_stage1/clean/config.json')
    pilot = load(run+'neural_stage1/pilot_state.json')
    statistics = load(run+'statistics/results.json')
    diagnostic = load(run+'paired_diagnostic/results.json')
    cpu = load(run+'cpu_diagnostics/results.json')
    if (sanity['status'], clean['status'], pilot['status'], statistics['status'], diagnostic['status']) != (
            'SANITY_PASSED', 'COMPLETED', 'COMPLETED', 'COMPLETED', 'COMPLETED'):
        raise RuntimeError('Expected completed input artifacts')
    metrics = clean['views']['clean']['metrics']
    scores = [(name.title(), statistics['results'][name]['views']['clean']) for name in ('frequency', 'markov')]
    scores.append(('Clean GRU', metrics))
    metric_rows = []
    for name, row in scores:
        budget = row['top_50']
        threshold = row['frozen_threshold']
        metric_rows.append(f"| {name} | {row['AP_all']:.5f} | {budget['tp']}/{row['positive_processes']} ({budget['recall']:.2%}) | {budget['precision']:.2%} | {threshold['tp']} / {threshold['fp']} / {threshold['fn']} | {budget['boundary_tied_processes']} |")
    cohort = clean['cohort']
    split = config['dataset']['split']
    labels = cpu['sibling_coverage']['counts']
    coverage = labels['annotated']
    path = cpu['path_novelty']
    rows = []
    for name, row in diagnostic['observations'].items():
        means = row['mean_NLL']
        distinct = row['block_distinct_input_counts']['dispersed']
        rows.append(f"| {name} | {row['effective_count']}/{row['targets']} ({row['effective_count']/row['targets']:.2%}) | {distinct}/{row['targets']} | {means['clean']:.5f} | {means['iid']:.5f} | {means['dispersed']:.5f} | {means['block']:.5f} | {row['block_minus']['dispersed']['all']['mean']:+.5f} |")
    report = f'''# Initial Experiment Results — 2026-11-18 (Asia/Shanghai)

**Decision:** Stop C03 expansion at M1. On this single THEIA campaign, clean type-only GRU scores rank attacks poorly against the shared controls. The paired long-outage diagnostic shows distortion of next-event prediction but does not establish a larger effect for blocks than for same-relation dispersion. C03's training benefit remains untested. C02 has data coverage for a future controlled experiment, not evidence of improved detection.

## Data and scope

- Public PIDSMaker DARPA E3 dumps and ORTHRUS/PIDSMaker author-curated process annotations. These are process annotations derived from DARPA reports, not event-level attack labels. No additional live attack data was generated.
- THEIA: 1,124,934,008-byte compressed dump, 44,366,117 parsed events, 96 of 96 annotated process UUIDs mapped across campaigns. CADETS: 1,465,153,615-byte dump, 36,484,667 events, 39 of 39 UUIDs mapped. Parsing retained all rows of those supplied dumps; standard PIDS configurations can apply different relation filters.
- Test: {split['test_start_utc']} through {split['test_end_utc']} (exclusive), encompassing the published 18:30–19:00 UTC Firefox campaign window. {cohort['original_target_events']:,} events, {metrics['processes']:,} original processes, {metrics['scoreable_processes']:,} actor-scored processes, {metrics['positive_scoreable']}/{metrics['positive_processes']} positives scoreable. The other original processes lack an actor score in this window. Apr03/04/05 train; Apr09 clean validation. CADETS data ready; neural evaluation deferred by the M1 gate.
- One seed (17), causal 30-minute past context limited to 128 events, embedding 32, GRU hidden 64, {config['parameters']:,} trainable parameters. 30,000 sampled benign events per epoch, 3 fixed epochs, Adam 1e-3, batch 16; checkpoint selected by clean validation CE. Scores are retrospective per-process maximum next-token NLL.

## R002 — full-window clean comparisons

| Model | AP_all | Recall@50 | Precision@50 | Frozen clean-val threshold TP / FP / FN | Number tied at top-50 cutoff |
|---|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

Here AP_all equals AP_observed because all 47 positives are scoreable. AP groups equal scores; top-50 ties follow deterministic node-index order. All models use the same target events and process universe. Published graph detectors are not part of this table. GRU validation CE fell across epochs ({', '.join(format(item['validation_CE'], '.4f') for item in clean['history'])}), showing successful learning on the prediction task; that did not translate into stronger detection. The frozen thresholds are model-specific clean-validation quantiles, not tuned on test labels.

## B1 — paired stress diagnostic using the frozen clean checkpoint

4096 label-blind sampled targets per split (seed 17). Mask seed 23, D=1799s from the benign-only amendment; the original 30/120s scheme rarely changed the model input. Each mode scores the same target; the three corrupt views match raw and effective deletion counts. Only event NLL is measured in this sampled diagnostic; no sampled process AP, simulated full-stream outage, or short outage claim.

| Split | Nonzero effective deletion | Block input differs from dispersed | Clean NLL | Uniform NLL | Dispersed NLL | Block NLL | Block minus dispersed |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Block increases average NLL versus clean; it has lower mean NLL than same-relation dispersion in both splits. The median paired difference versus dispersion is zero across all targets. Differences are descriptive; events share actors and contexts. This does not measure detection changes or rule out a benefit from augmentation training.

## CPU signal and C02 coverage

- Retrospective exact executable path allowlist: {path['seen_paths']} paths from training-day actors; {path['novel_test_actors']} test actors use previously unseen paths, including {path['novel_annotated_actors']} of 47 annotated attack actors. AP_all {path['metrics']['AP_all']:.5f}. Binary novelty has broad ties; the static subject path's online availability is not established. This is a diagnostic that the dump carries useful semantic information, not a fair type-only or prospective detector baseline.
- Observed clone/fork links in the indexed days, grouped by parent process instance: {coverage['processes_with_prior_sibling']}/{coverage['processes']} annotated actor processes have at least one target preceded by observable sibling activity; {labels['unannotated']['processes_with_prior_sibling']}/{labels['unannotated']['processes']} unannotated actor processes do too. This is stricter than counting siblings appearing anywhere later in the window. Incomplete logs can hide links. No sibling model or alternative-context control has been trained. A C02 pilot would need a parent-only control, a matched non-sibling context control, and causal path semantics before adding semantic inputs.

## M0 resource feasibility and reproducibility

- RTX 3060 Laptop 6 GB with Torch {config['torch']} / CUDA build {config['cuda']}. 256-target real-data sanity passed. Clean training {clean['training_seconds']:.2f}s; the full clean run took {clean['total_wall_seconds']:.2f}s, including {clean['views']['clean']['inference']['seconds']:.2f}s for cached test inference. The serial sanity and clean jobs used {pilot['gpu_wall_seconds']:.2f}s wall time; frozen-checkpoint diagnostic added {diagnostic['elapsed_seconds']:.2f}s inclusive of CPU mask construction. These are elapsed GPU-job times, not the device's busy time.
- PyTorch peak allocated {clean['resources']['torch_cuda_allocated_peak_MiB']:.2f} MiB, reserved {clean['resources']['torch_cuda_reserved_peak_MiB']:.2f} MiB, sampled process RSS {clean['resources']['process_RSS_peak_MiB']:.2f} MiB. Driver context and other processes are excluded from allocator figures. No claim of end-to-end online latency or fit of a full GNN.
- Exact sanity/clean commands and timestamps: `{run}neural_stage1/pilot_state.json`. Checkpoint, loss log, configuration, process scores and NLL: `{run}neural_stage1/clean/`. Other machine-readable results: `{run}statistics/results.json`, `{run}paired_diagnostic/results.json`, `{run}cpu_diagnostics/results.json`. Review: Round 1 raw report in `EXPERIMENT_CODE_REVIEW_round1.md`; partial Round 2 checks in `.aris/traces/experiment-bridge/round2.partial.md`.

**Gate disposition:** R001 complete; R002 clean comparisons and sampled B1 diagnostic complete (full-window corrupt detection metrics not run); R003–R008 remain pending or deferred. Only C02's B002 CPU coverage check ran. PRO-TRAP, ORTHRUS and MAGIC remain unreproduced. No main-method benefit, multiple-campaign generalization, or submission readiness is claimed.
'''
    publish('refine-logs/EXPERIMENT_RESULTS.md', report, 'Real THEIA results, resources and M1 gate disposition')
    track = '''# Experiment Tracker — 2026-11-18 (Asia/Shanghai)

| Run | Milestone | Status | Evidence / Next gate |
|---|---|---|---|
| P000 | M0 synthetic resource probe | COMPLETE_RESOURCE_ONLY | Historical shape test, separate from real data |
| R001 | M0 data, labels and real-data sanity | COMPLETE | Both datasets downloaded/indexed for pilot; THEIA CUDA sanity passed |
| R002 | M1 frequency/Markov/clean GRU and B1 | COMPLETE_PILOT_GATE_NOT_MET | All clean cohort metrics and sampled long-outage NLL. No full-window corrupt ranking. See EXPERIMENT_RESULTS.md |
| R003 | M2 clean/iid/dispersed/block training | DEFERRED_M1 | Only clean GRU completed; weak clean detection and insufficient supporting B1 evidence |
| R004 | M3 temporal crop | DEFERRED_M1 | Inspired control, no full PRO-TRAP claim |
| R005 | M3 CADETS pilot | DATA_READY_TRAINING_DEFERRED | Data and indexes available; plan requires first-domain gate |
| R006 | M4 two domains/three seeds | DEFERRED_M1 | No multi-seed evidence |
| R007 | M4 published PIDS | NOT_RUN | Official implementations identified; no same-cohort reproduction |
| R008 | M4 unseen duration/pure time/full stream/resources | NOT_RUN | Offline target-conditional stress evidence only |
| B002 | C02 CPU coverage | COMPLETE_COVERAGE_ONLY | 45/47 annotated actors with observable prior sibling activity |
| B006 | C06 write-first-exec diagnosis | NOT_RUN | Remains a backup |
'''
    publish('refine-logs/EXPERIMENT_TRACKER.md', track, 'Pilot execution tracker and deferred milestones')
    state = load('refine-logs/BRIDGE_STATE.json')
    state.update(stage='M1_COMPLETE_GATE_NOT_MET', report='refine-logs/EXPERIMENT_RESULTS.md',
                 gpu_runs_launched=3, gpu_training_jobs=2, gpu_frozen_diagnostic_jobs=1,
                 C03_augmentation_effect='UNTESTED', C02='CPU_COVERAGE_ONLY',
                 paper_ready=False, test_clean_AP_all={name: row['AP_all'] for name, row in scores})
    publish('refine-logs/BRIDGE_STATE.json', json.dumps(state, indent=2), 'Machine-readable final pilot state')
    contract_file = 'idea-stage/docs/research_contract.md'
    contract = (ROOT / contract_file).read_text(encoding='utf-8').split('\n## 真实 Pilot 执行更新')[0]
    update = '''## 真实 Pilot 执行更新（2026-11-18）

C03 已执行 THEIA 真实数据的频次、一阶转移、clean GRU 及冻结模型配对 NLL 诊断。clean GRU AP 0.01671，Recall@50 为 1/47；同数据、同评分总体的一阶转移 AP 0.02215。训练可运行，但检测与缺失形状证据未通过 M1 扩跑门槛。尚未测量块训练的增强收益。报告见 `refine-logs/EXPERIMENT_RESULTS.md`。CADETS 已准备、未训练；近期已发表 PIDS 未复现。

C02 的 CPU 覆盖初检：47 个标注、可评分的攻击进程中 45 个具备先前同父兄弟活动。覆盖率仅表明该方向可构造输入；仍需父进程单独输入和匹配非兄弟对照来判断信息增益。暂不改写研究主张。
'''
    publish(contract_file, contract.rstrip()+'\n\n'+update, 'Research contract linked to measured pilot evidence')


if __name__ == '__main__':
    main()
