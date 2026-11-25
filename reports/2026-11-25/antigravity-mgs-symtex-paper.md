# Antigravity 独立研究线：MGS-SymTex 跨场景柑橘病征识别小论文完整实验与外部封口盲测

> 日期：2026-11-25  
> Agent/任务：Antigravity / 独立方法筛选与 SCI 小论文收官实验归档（与 GPT、DeepSeek 成果分目录隔离存放）  
> 对应目录：`antigravity/`  
> 分支：main

## 任务

围绕跨场景柑橘病害症状识别的 SCI 小论文目标，在不触碰、不覆盖 GPT 与 DeepSeek（DS）既有文档及中央协调状态（`coordination/`、`docs/experiments/`）的前提下，独立完成候选创新路线筛选、`MGS-SymTex`（颈部流形马氏选择性级联 + 零推理增耗病理双先验特征塑形）全 3 随机种子消融与端到端检测评测、RTX 4060 Ti 实测单图延迟测定、以及 4 个像素去重独立外部公开数据集的一次性零样本封口盲测，并将代码、轻量指标 JSON、300 DPI 论文配图、权重 SHA-256 清单及定稿报告集中归档至独立目录 `antigravity/`。

## 实际完成

- 完成三路线预注册筛选（`Probe-A1 / A2 / B0 / C0`），如实归档阴性/受限路线（单阶段框内保护光测干预+PCGrad `Probe-B0` 源域 `val` `-1.79pt` 停止；跨源零样本开放集拒识 `Probe-A2` 同域未知虫害混淆停止），锁定通过 3 种子门槛的 `MGS` 选择性级联与 `SymTex` 病理双先验塑形。
- 完成实验 E1（`exp1_symtex_ablation_and_kd.py`）：在统一 14 类 `SciDB` 协议下完成 6 组组件消融与蒸馏变体 × 3 随机种子（`0, 1, 2`）= 18 个 `3.347M` 剪枝模型训练，验证推理期剥离 `BASD` 与 `II-HFR` 全部辅助分支实现零额外参数/零额外 FLOPs 部署，且无教师 `SymTex_Pruned` 取得 `testB 82.35±1.97%`、`testC 82.70±1.13%`（黄化簇 `84.36±1.98%`，较基线 `+2.19pt`）及 `CitrusUAT` 叶片掩膜能量占比 `68.53±0.25%`（`+BASD` 单项达 `70.00±0.44%`）。
- 完成实验 E2（`exp2_end2end_mgs_cascade_and_timing.py`）：在同进程严格 FP32 方形 640 `COCO AP50-95 / AP50` 口径下完成对角线配对 3 种子端到端评测与 RTX 4060 Ti 单图实测计时。量化揭示 100% 无条件轻量级联因剥离全局上下文导致 `val / testB` 掉点 `-1.45pt / -4.63pt` 且耗时翻倍（`8.35 ms → 16.40 ms`）；仅按源域 `val` 标定 `q=0.80` 冻结阈值的 `MGS` 在源域仅触发 `20.0%` 候选框，跨场景自动抬升至 `48.2% / 44.9%`，使端侧轻量档 `testC AP50-95` 提升 `+4.36pt ~ +4.53pt`，高精度档 `Ours-Pro (MGS + DINOv2)` 在 `val / testB / testC` 达到 `90.12±0.60%` / `74.59±2.60%`（`+3.34pt`）/ `81.68±1.36%`（`+8.91pt`），源域单图耗时仅 `12.10 ms`（约 82.6 FPS）。
- 完成实验 E3（`exp3_external_blind_validation.py`）：完全冻结模型权重与源域 `val q=0.80` 阈值，在 4 个与 `SciDB` 零像素重叠的独立公开数据集上执行一次性零样本封口盲测。在 `Zenodo 11148332` 真实果园 519 图 / 1,765 个归一化核验叶框（健康叶 vs 斑驳黄化疑似 vs 缺素对称黄化疑似）上，3 类病理拓扑框分类宏 F1 由单流基线 `56.67±4.30%`（纯蒸馏 `58.71±1.58%`）提升至无教师 `SymTex_Pruned` `59.05±3.03%` 与联合版 `Ours_SymTex_KD_Pruned` `60.82±2.04%`（`+4.15pt`）；在 `Sweet Orange V1`（`N=1,135`）上提升至 `77.07±4.66%`（`+4.43pt`）；在跨树种柚叶 `Pomelo V2`（`N=292`）上，`+BASD` 将宏 F1 由 `63.33%` 提升至 `68.67±1.74%`（`+5.34pt`）。
- 完成实验 E4（`exp4_generate_paper_figures.py`）与论文定稿总报告 `antigravity/docs/00_SCI小论文完整实验结果与写作定稿.md`，导出 3 张 300 DPI 零遮挡插图及 19 个本地权重的 `MODEL_CHECKPOINTS_MANIFEST.csv`。

## 证据与来源

- 环境与硬件：Windows 11，NVIDIA RTX 4060 Ti 16GB，`D:\anaconda\envs\minidector\python.exe`（`torch 2.5.1+cu121`，`torchvision 0.20.1+cu121`，`ultralytics 8.3.245`，`faster_coco_eval`，严格关闭 TF32 与 cuDNN benchmark）。
- 源域与跨背景诊断数据：统一清洗去漏 14 类 `SciDB` 划分（`train 7,136 / val 866 / testB 602 / testC 602`），全部超参数、最佳轮次与路由分位数 `q=0.80` 仅在 `SciDB val` 上选定。
- 独立外部盲测数据源：严格读取本地已像素去重审计的 `D:\病虫害论文\新增公开数据_20260916\中文分类与标签使用索引.json`（`索引去重保留=True` 且 `同库跨标签冲突=False`），涵盖 `Zenodo 11148332`（CC BY 4.0，果园框包 519 图 / 1,765 框及三类包 560 图）、`Sweet Orange V1`（`f7cr74mwpj`，CC BY 4.0，1,135 图）、`Pomelo V2`（`vk7dnf8nrj`，CC BY 4.0，292 图）与 `CitrusUAT`（Zenodo 8294078，CC BY 4.0，953 图）。遵守病理边界，所有黄化框与图像均按视觉症状报告，不宣称 PCR 确诊黄龙病；190 张无专家真值自采胡柚图未纳入定量准确率统计。
- 结果摘要文件：`antigravity/results/e1_symtex_ablation_and_kd_summary.json`、`e2_end2end_mgs_symtex_summary.json`、`e3_external_blind_validation_summary.json` 及 `probe_*.json`；本地 `.pt` 权重 `SHA-256` 清单见 `antigravity/MODEL_CHECKPOINTS_MANIFEST.csv`。

## 修改文件

- 新增 `antigravity/README.md`
- 新增 `antigravity/MODEL_CHECKPOINTS_MANIFEST.csv`
- 新增 `antigravity/docs/00_SCI小论文完整实验结果与写作定稿.md`
- 新增 `antigravity/docs/01_总体设计与三路线筛选协议.md`
- 新增 `antigravity/docs/02_三路线实测裁决与实施方案.md`
- 新增 `antigravity/docs/实验记录_小论文完整实验_20261125.md`
- 新增 `antigravity/docs/实验记录_三路线创新筛选_20261124.md`
- 新增 `antigravity/scripts/paper_suite/exp1_symtex_ablation_and_kd.py`
- 新增 `antigravity/scripts/paper_suite/exp2_end2end_mgs_cascade_and_timing.py`
- 新增 `antigravity/scripts/paper_suite/exp3_external_blind_validation.py`
- 新增 `antigravity/scripts/paper_suite/exp4_generate_paper_figures.py`
- 新增 `antigravity/scripts/screening/probe_a1_selective_routing.py`
- 新增 `antigravity/scripts/screening/probe_a2_hierarchical_edl.py`
- 新增 `antigravity/scripts/screening/probe_a2_domain_debiased_osr.py`
- 新增 `antigravity/scripts/screening/probe_b0_context_ortho_det.py`
- 新增 `antigravity/scripts/screening/probe_c0_pathology_dual_stream.py`
- 新增 `antigravity/results/e1_symtex_ablation_and_kd_summary.json`
- 新增 `antigravity/results/e2_end2end_mgs_symtex_summary.json`
- 新增 `antigravity/results/e3_external_blind_validation_summary.json`
- 新增 `antigravity/results/probe_a1_selective_routing_summary.json`
- 新增 `antigravity/results/probe_a2_hierarchical_edl_summary.json`
- 新增 `antigravity/results/probe_a2_plus_debiased_osr_summary.json`
- 新增 `antigravity/results/probe_b0_ctx_ortho_summary.json`
- 新增 `antigravity/results/pcgrad_training_conflict_stats.json`
- 新增 `antigravity/results/probe_c0_symtex_summary.json`
- 新增 `antigravity/figures/fig1_pareto_ap_latency.png`
- 新增 `antigravity/figures/fig2_symtex_ablation_and_external.png`
- 新增 `antigravity/figures/fig3_external_orchard_blind_det.png`
- 新增 `reports/2026-11-25/antigravity-mgs-symtex-paper.md`（本报告）
- 修改 `README.md`（在目录列表增加 `antigravity/` 独立归档入口说明，未改动其他段落）

## 验证命令与结果

```powershell
D:\anaconda\envs\minidector\python.exe scripts/validate_repository.py
git status --short
```

- `scripts/validate_repository.py` 全量校验与变更报告校验均通过（零禁用扩展名 `.pt/.zip/.docx/.pdf`、零禁用目录名 `checkpoints/weights/runs/datasets`、单文件最大 428 KB 远低于 20 MiB 上限、报告章节齐全）。
- 未修改 `coordination/PROJECT_STATE.md`、`coordination/DECISIONS.md`、`coordination/CURRENT_ROUND.md` 及任何 GPT / DeepSeek 历史报告与文档。

## 未完成或阻塞

- 常山胡柚 / 香柚自采 190 张照片的植保专家盲评尚未回填，公司果园数据尚未接入，因此目标品种胡柚上的定量准确率验证仍待真值到位后补测。
- 跨采集源零样本开放集未知虫害拒识（`Probe-A2`）在无目标域校准样本时存在同域背景偏移与类别不平衡干扰，作为论文 Discussion 中的未来工作说明，不纳入本文创新主张。

## 对下游的影响

- 为 SCI 小论文提供了完整的 3 随机种子主表、正交消融表、RTX 4060 Ti 实测延迟帕累托数据、四源独立公开集零样本盲测证据及 300 DPI 配图，可直接进入英文初稿撰写。
- 全部产物隔离在 `antigravity/` 下，不影响协调者在 `coordination/` 中维护的既有任务流与 DeepSeek 归档记录。

## 下一步建议

1. 基于 `antigravity/docs/00_SCI小论文完整实验结果与写作定稿.md` 与 `antigravity/figures/` 起草英文 SCI 论文正文（LaTeX 或 Word）。
2. 待 16 组胡柚专家盲评包或合作方胡柚真值回填后，直接加载 `antigravity/MODEL_CHECKPOINTS_MANIFEST.csv` 所列冻结权重补跑胡柚外部验证表。
