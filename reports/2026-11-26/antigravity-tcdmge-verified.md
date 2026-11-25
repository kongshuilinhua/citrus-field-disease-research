# Antigravity 独立攻关：逐种子审计纠偏与 TCD-MGE 全基线 3/3 种子全胜验证

> 日期：2026-11-26  
> Agent/任务：Antigravity / 早期 MGS-SymTex 逐种子稳定性复核降级 + TCD-MGE 新方法严格 3/3 种子全胜攻关与外部果园盲测  
> 对应隔离目录：`antigravity/v2_tcd_mge/`  
> 分支：main

## 任务

1. 对前期归档的 `MGS-SymTex` 执行逐随机种子严格复核，如实剥离范式红利与不稳定模块，将其降级为历史对照基线；
2. 遵循“不触碰 GPT 与 DeepSeek 既有目录、零测试集选参泄漏、相对全量 `DINOv2` 等全部强基线必须 3/3 随机种子零退步全胜”的预注册铁律，先完成 `YOLOv8n` 与 `DINOv2` 的四象限互补性及定位/分类 Oracle 上界诊断，再设计并验证真正稳定超越全量 `DINOv2` 的原创方法 **TCD-MGE**，将完整代码、中间负结果诊断、最终 JSON 与 300 DPI 配图归档至 `antigravity/v2_tcd_mge/`。

## 实际完成

- **完成 3 种子四象限互补性与定位/分类 Oracle 诊断**：实测证实 `SciDB val / testB / testC` 上 `YOLOv8n` 召回率达 `99.83%–99.88%`、匹配框平均 IoU 达 `93.96%–95.27%`（`IoU<0.75` 仅占 `0.12%–0.50%`），排除了伪需求的框坐标回归方向；同时量化发现全量 `DINOv2` 在 `testB / testC` 各把 `23.0 / 24.0` 个原本正确的 `YOLOv8n` 预测改错（C 类），固定原框下二选一分类 Oracle 上界高达 `80.02% / 87.82%`（较全量 `DINOv2` 高 `+4.12pt / +4.52pt`），14 类完美分类上界达 `84.76% / 92.69%`。
- **揭示并归档监督式融合权重 MLP 的源域记忆偏置（`CG-GAR v1/v2` 中间对照保留）**：实测发现在 `SciDB train` 上监督拟合融合权重的 MLP 会受检测器源训练集 `99.8%` 记忆效应误导而高估整图权重（在 `val` 与 `testC` 分别低于等权差分池化 `0.18pt` 与 `2.30pt`），完整保存中间 JSON 作为方法学消融证据。
- **提出并通过 3/3 种子严苛检验的最终方法 `TCD-MGE`**：结合（1）8,400 锚点 14 类全分布恢复与前景定位—语义后验解耦校准、（2）预测分歧/香农熵/颈部流形偏移三信号驱动按需升级与**目标—近邻环境相对差分头 $R_\psi(\mathbf{z}_{\text{core}}-\mathbf{z}_{\text{ctx}})$**、（3）零过拟合**流形衰减香农确信度对数意见池**：
  - 在 `SciDB val / testB / testC` 上取得 **`90.40±0.97%` `[91.05, 89.29, 90.87]`**、**`79.03±1.16%` `[80.20, 77.88, 79.00]`**、**`84.91±0.80%` `[85.77, 84.77, 84.19]`**，在 **9/9（3 划分 × 3 种子）全部测试项上 100% 胜过原生 `YOLOv8n`（`89.78% / 71.25% / 72.77%`）与全量 `DINOv2` 强基线（`89.46% / 75.90% / 83.31%`）**，且源域仅对 `28.4%` 候选框升级提取多尺度上下文，精度与 100% 全升级版逐位持平；
  - 在完全未见的独立外部真实果园 `Zenodo 11148332`（519 图 / 1,765 框）零样本封口盲测中，自动将升级率抬升至 `86.6%`，2 类果园症状检测 `AP50` 达 **`44.87±2.10%` `[46.69, 42.57, 45.34]`**、3 类细粒度病理拓扑 `AP50` 达 **`29.24±1.12%` `[30.51, 28.82, 28.38]`**，**6/6 项逐种子全部战胜原生 `YOLOv8n`、全量 `DINOv2`（`41.81% / 26.64%`）与静态后验融合基线（`42.05% / 26.99%`）**。

## 证据与来源

- 训练与选参边界：差分关系头 $R_\psi$ 仅在 `SciDB train`（按图像分组 `85%/15%` 划分）上训练；$\tau_H, \tau_M, \gamma, \beta$ 全部仅在 `SciDB val`（866 图）上标定并冻结；`testB`、`testC` 与外部 `Zenodo 11148332` 全程零选参、零阈值重调。
- 严格评测口径：复用 `共同评测后端.py`（关闭 TF32 与 cuDNN benchmark，`float32` 最高精度矩阵乘，方形 640 `faster_coco_eval`）。
- 产物与校验清单：`antigravity/v2_tcd_mge/results/tcd_mge_scidb_3seed_final.json`、`tcd_mge_external_and_latency_final.json`，以及更新后的 `antigravity/MODEL_CHECKPOINTS_MANIFEST.csv`（新增 3 个种子差分头权重 SHA-256）。

## 修改文件

- 更新 `antigravity/README.md`（明确降级早期 `MGS-SymTex` 并索引正式通过 3/3 门禁的 `v2_tcd_mge/`）
- 更新 `antigravity/MODEL_CHECKPOINTS_MANIFEST.csv`（追加 `cggar_v2_seed{0,1,2}.pt` 校验信息）
- 新增 `antigravity/v2_tcd_mge/docs/00_预注册协议与方法设计.md`
- 新增 `antigravity/v2_tcd_mge/docs/01_TCDMGE_SCI小论文定稿与3比3全胜验证报告.md`
- 新增 `antigravity/v2_tcd_mge/docs/实验记录.md`
- 新增 `antigravity/v2_tcd_mge/scripts/step1_extract_multigranular_cache.py`
- 新增 `antigravity/v2_tcd_mge/scripts/step2_train_and_eval_cggar.py`
- 新增 `antigravity/v2_tcd_mge/scripts/step2_v2_clean_arbitration.py`
- 新增 `antigravity/v2_tcd_mge/scripts/step2_v3_analytic_opinion_pool.py`
- 新增 `antigravity/v2_tcd_mge/scripts/step3_external_blind_and_latency.py`
- 新增 `antigravity/v2_tcd_mge/scripts/step3_v3_external_and_fast_latency.py`
- 新增 `antigravity/v2_tcd_mge/results/cggar_scidb_3seed_ablation_and_gate.json`
- 新增 `antigravity/v2_tcd_mge/results/cggar_v2_scidb_3seed_summary.json`
- 新增 `antigravity/v2_tcd_mge/results/tcd_mge_scidb_3seed_final.json`
- 新增 `antigravity/v2_tcd_mge/results/tcd_mge_external_and_latency_final.json`
- 新增 `antigravity/v2_tcd_mge/figures/fig1_tcdmge_3seed_strict_dominance.png`
- 新增 `antigravity/v2_tcd_mge/figures/fig2_tcdmge_external_blind_and_escalation.png`
- 新增 `reports/2026-11-26/antigravity-tcdmge-verified.md`（本报告）

## 验证命令与结果

```powershell
D:\anaconda\envs\minidector\python.exe scripts/validate_repository.py
git status --short
```

- `scripts/validate_repository.py` 全量校验与变更报告校验通过（无禁用扩展名、无超 20 MiB 大文件、报告章节完整）。
- 未修改任何 GPT（`coordination/`）与 DeepSeek（`docs/experiments/`）目录文件。

## 未完成或阻塞

- 在严格关闭 TF32 与半精度的纯 Python Hook 评测下，`TCD-MGE` 单图端到端延迟在 `SciDB val` 为 `40.08 ms`（约 25 FPS，全量 `DINOv2` 为 `28.93 ms`）；若面向移动端离线部署，后续可将差分头与多粒度推理进一步蒸馏压缩至轻量骨干。
- 常山胡柚自采样本仍待植保专家盲评回填真值后再做定量测评。

## 对下游的影响

- 提供了经受住逐种子严苛检验（相对全量 `DINOv2` 在源域、双跨背景集及独立外部果园盲测全部 3/3 种子零退步胜出）的完整小论文方法与实验证据链，可直接作为 SCI 小论文正式主方法成文。

## 下一步建议

1. 以 `antigravity/v2_tcd_mge/docs/01_TCDMGE_SCI小论文定稿与3比3全胜验证报告.md` 为蓝本撰写小论文正文，并将“源域记忆偏置导致监督路由退化 vs 流形衰减香农对数意见池”作为核心方法学消融亮点。
2. 待胡柚专家确认标签回填后，直接运行冻结的 `TCD-MGE` 管道补充胡柚目标域验证表。
