# 审计日志缺失下的进程异常排名

本项目包含 THEIA、CADETS 两组离线审计数据的检测实验，以及论文图表和 LNCS 稿件。当前论文正文为 4 张表、10 张图。原始数据、训练模型、派生结果和论文图件各自独立存放。

## 快速使用

在项目根目录运行：

```powershell
python reproduce.py check
python reproduce.py figures
python reproduce.py paper
python reproduce.py paper --redraw --output build/paper-from-results
```

| 命令 | 用途 | 需要 |
|---|---|---|
| `check` | 运行既有的历史窗口、指标计算和删失配对测试 | Python、NumPy |
| `figures` | 从冻结结果 CSV 再生图和逐图源数据 | Python、Matplotlib、NumPy、Pillow |
| `paper` | 使用当前 TeX 和已有矢量图编译正文与数值补充 | latexmk、pdfLaTeX、BibTeX、LNCS 类及论文宏包 |
| `paper --redraw` | 在独立目录先再生图，再编译论文 | 上述绘图与编译环境 |
| `bundle` | 打包代码、稿件源码、矢量图与绘图所需的冻结结果 | Python 标准库 |

生成文件写入新目录。重复执行时用 `--output` 指定另一个目录；已有目录不会被覆盖。默认出图产物在 `build/figures/runs/manuscript_figures_v2/`，默认 PDF 在 `build/manuscript/paper/`。用 `python reproduce.py --help` 查看入口。

使用 `--redraw` 后可用原 QA 环境复核重建产物，例如：

```powershell
py -3.11 build/paper-from-results/paper/.aris/verify_manuscript_figures.py
```

该检查核对图源与稿件、面板排版、关键 AP 和告警计数。执行 `figures` 或 `paper --redraw` 仅使用冻结的结果表，不训练模型、不需要 CUDA 或原始审计日志。

## 代码分区

| 路径 | 内容 |
|---|---|
| [experiments/](experiments/README.md) | 数据处理、模型与基线、融合、补充实验、结果汇总 |
| `data/` | 原始导出、处理后按日事件、训练/测试缓存 |
| `runs/` | 每次实验及汇总结果、论文绘图输入 |
| [paper/](paper/README.md) | 论文、图表构建器、核查脚本及实验协议 |
| `refine-logs/` | 实验记录、完整结果及设计变更说明 |
| `external/` | PIDSMaker 和 Pro-Trap 的固定上游版本 |

绘图环境见 [requirements-figures.txt](requirements-figures.txt)。训练和旧模型读取依赖记在 [实测环境](experiments/reproduced-environment.txt)。代码包提供复画和编译所需的文件；从头训练需另备数据、外部组件和训练环境，具体入口见实验说明。
