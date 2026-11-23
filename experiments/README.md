# 实验代码导航

从项目根目录调用 `python experiments/<script>.py --help` 查看支持参数。保留脚本原位置，因为训练、评估、报告间使用同目录导入，且不同阶段读取已有的运行目录。本页区分论文使用的正式流程与早期验证。

| 阶段 | 入口与用途 | 输入 / 产物 |
|---|---|---|
| 读取归档 | `download_inputs.py`、`setup_pg_client.py`、`prepare_dataset.py` | PostgreSQL dump 和 PIDSMaker 注释 → `data/processed/<source>_e3/` |
| 建索引与上下文缓存 | `index_events.py` → `build_caches.py` → `add_stress_views.py`、`train_bank.py` | 按日事件 → 固定样本、token 字典、评估视图及训练上下文 |
| 检测模型与基线 | `local_grid.py`（主队列）、`train_eval.py`（单个 GRU），`count_baseline.py`、`protrap_adapted.py`（单独运行），`heldout_campaigns.py`（额外窗口） | 缓存 → `runs/full_grid_v1/` 中的模型和指标；队列自动运行统计基线及 MAGIC 适配版 |
| 路径与结构融合 | `semantic_fusion.py` → `structural_fusion.py` → `report_semantic_fusion.py` | 冻结 GRU、Count IF、路径及结构证据 → `runs/semantic_fusion_v1/` 和 `runs/semantic_fusion_v2/comparison.csv` |
| 配对消融和缺失重放 | `fusion_sensitivity.py`、`outage_replay.py` | 已拟合模型及原进程集合 → `runs/fusion_sensitivity_v1/`、`outage_replay_v1/`、`outage_additional_v1/` |
| 补充聚合与告警实验 | `method_benign.py` → `method_followup.py` → `evidence_aggregation.py` | 分离的正常窗口及冻结预测 → `runs/method_followup_v1/`、`runs/evidence_aggregation_v1/` |
| 指标与验数 | `evaluation.py`、`test_correctness.py`、`paper/.aris/verify_*.py` | 固定原始进程队列；无分数正例仍计入 AP/召回分母 |
| 试点与辅助诊断 | `pilot_driver.py`、`report_pilot.py`、`diagnose_signal.py`、`check_train_masks.py`、`fast_paired_cache.py`、`paired_stress_check.py`、`profile_grid.py` | 历史试点、训练缓存诊断和资源测量，复画论文时不需要 |

## 从现有工作区复查实验

- 完整训练的缓存名为 `theia_longoutage_v1`、`cadets_longoutage_v1`。运行前查看各缓存 `split.json` 和 `dataset.json`，并查 `runs/full_grid_v1/grid_state.json` 中记录的参数。`build_caches.py` 和 `train_eval.py` 的默认值含早期试点设置，不能直接当成正式实验配置。
- 正式 GRU 网格为种子 17、23、42，限制在单张 RTX 3060 Laptop GPU。`local_grid.py` 是续跑队列；其默认输出位置为现有正式目录。跑新实验时可用独立 `--output`；后续读取固定路径的脚本需要在同一份隔离工作区运行，避免误读旧结果。
- 论文图表从已有指标再生。冻结结果的对应关系、校准范围及重复观测说明，分别见 [聚合实验协议](../paper/METHOD_EVALUATION_PROTOCOL.md) 和 [告警迁移协议](../paper/EVIDENCE_AGGREGATION_PROTOCOL.md)。完整复跑需要原始数据、外部版本和模型环境；轻量代码包不附原始数据和模型权重。
- Count、Structure、Path 的拟合或聚合与 GRU 推理是分开的。复用冻结预测时应保留原来的 train/validation/test 划分和可评分进程口径；不要将 35 个条件视为 35 个独立攻防窗口。
