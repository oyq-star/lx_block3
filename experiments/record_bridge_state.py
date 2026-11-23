"""Versioned execution notes; does not change the original scientific plan."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'idea-stage/evidence'))
from artifact_io import write_artifact

note='''# Pilot execution amendment — 2026-11-17

This is an exploratory, train-calibrated implementation amendment to the one-time duration gate in FINAL_PROPOSAL.md, paragraph 39. No detector test performance has been computed at freeze time.

- Original 30/120s mask: effective input changes on 3/1000, 10/1000, 20/1000 benign THEIA targets (0.3%, 1.0%, 2.0%). The current recent128 input usually spans only fractions of a second; uniform start over30min misses it.
- A single train-only duration sweep is saved in TRAIN_MASK_DIAGNOSTIC.json (seed17, mask23, 1000 targets/day). Candidate tuples were written in the script before execution. We inspected all results; this is calibration, not a preregistered statistical test.
- Freeze train D=(1790,1799)s, test D=1799s for the initial revised stress pilot. This is the shortest tested tuple yielding at least10% nondegenerate effective changes on each THEIA training day (12.4%,18.3%,31.9%). These decision criteria were chosen during this exploratory calibration, before any detector test metric. No mask placement, relation sampling, context, loss, or model change.
- D=1800 is rejected: all active relation-dispersed masks degenerate. CADETS checks are descriptive only; its run remains gated by first-domain results.
- This tests approximately30-minute relation outages. It cannot support a claim about30/120s outages or realistic source outages. Original failure is retained. No further adjustment based on attack results is permitted.
- Fresh cache data/cache/theia_longoutage_v1; old cache metadata cannot be mixed. Clean baseline and short sanity first. Four-variant training only if M1 has detection and nontrivial shape signal; otherwise preserve the failed/inconclusive gate.
- Split: train available UTC Apr03/04/05; clean validation Apr09 18–20UTC; test Apr10 18–20UTC, enclosing the full published18:30–19UTC campaign. Original process universe includes actor and process objects; labels are author-curated ORTHRUS/PIDSMaker process UUID annotations derived from DARPA.
- One-seed pilot: embedding32,GRU64,K128,batch16,Adam1e-3,3fixed epochs, two forward calls even for clean. Best checkpoint by clean-val CE; patience is diagnostic only to preserve equal steps.
- pilot_driver.py enforces local serial execution,7200s aggregate wall time including sanity/scoring, never more than10800s. train_eval.py also checks its deadline in each evaluation batch. Point-in-time GPU and allocator memory are not an online throughput guarantee.
- Exact served reviewer model variant is unavailable; native Codex route requested gpt-6-astra/xhigh, self-reported Codex/GPT-6 family. Treat as same-family provisional.
'''
write_artifact('refine-logs/PILOT_EXECUTION_AMENDMENT.md',note,'Train-only one-time mask duration calibration',skill='experiment-bridge',stage='implementation')
state=dict(stage='DATA_READY_CODE_REREVIEW_PENDING',datasets={},gpu_runs_launched=0,protocol='refine-logs/PILOT_EXECUTION_AMENDMENT.md')
for name in ('theia','cadets'):
    root=ROOT/('data/processed/'+name+'_e3')
    state['datasets'][name]=dict(schema=json.loads((root/'schema.json').read_text()),
                                compressed_dump_bytes=(ROOT/('data/downloads/'+name+'_e3.dump')).stat().st_size,
                                extraction_complete=(root/'EXTRACTION_COMPLETE').exists())
write_artifact('refine-logs/BRIDGE_STATE.json',json.dumps(state,indent=2),'Actual data-preparation and launch state',skill='experiment-bridge',stage='execution')

review=ROOT/'refine-logs/EXPERIMENT_CODE_REVIEW_round1.md'
write_artifact('refine-logs/EXPERIMENT_CODE_REVIEW.md',review.read_text(encoding='utf-8'),'Fresh code-review raw response round1',skill='experiment-bridge',stage='review')
trace=ROOT/'.aris/traces/experiment-bridge';trace.mkdir(parents=True,exist_ok=True)
(trace/'round1.response.md').write_bytes(review.read_bytes())
(trace/'round1.meta.json').write_text(json.dumps(dict(transport='native-spawn_agent',backend='codex',requested_model='gpt-6-astra',reasoning_effort='xhigh',actual_exact_variant=None,reported_family='Codex/GPT-6',assessment='same-family provisional'),indent=2),encoding='utf-8')
print('Execution amendment, state, and code-review trace saved.')
if __name__=='__main__':pass
