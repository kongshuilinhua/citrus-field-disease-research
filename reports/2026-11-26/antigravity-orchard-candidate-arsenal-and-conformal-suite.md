# Antigravity 预注册两层八候选果园方法库与有限样本保形风险控制实测报告

## 任务
围绕用户“路线 1（稳定超越基线的零作假检测校准候选）与路线 2（保形风险控制与分层复核候选）并行储备，待真实果园数据到位后提供充分候选选择”的需求，严格执行：
1. 归档前置只读可行性审计结果（量化源域冲突饥饿、分解式 Oracle 天花板、冲突可预测性 AUC 置信区间与类级收缩效应），明确早停复杂源域逐框学习门控的实证依据；
2. 在不重训基座权重（强制核验全部 6 份冻结权重 `SHA-256`）、不接触任何测试集选参（全部标量超参数仅在 `SciDB val` 锁定）的前提下，构建并评测第一层 5 个梯度互补的零样本检测校准候选（`C1–C5`），补充面向胡柚手机 App 的 `Top-1/2/3` 交互式症状候选命中率与非生物缺素胁迫 OOD 区分指标；
3. 在严格“校准集与评测集 100% 图像互斥（200 次独立随机划分）”协议下，实现并评测第二层 3 个选择性复核与保形风险控制候选（`R1` 零标签视觉纠错送审、`R2` 有限样本保形风险控制 CRC、`R3` 软后验经验贝叶斯 `597` 例 PCR 疑似症状分层送检），并提供真实果园即插即用 CLI 工具与选型手册。

## 实际完成
1. **完成前置只读可行性审计归档（解释为何早停源域学习门控）**：
   - 在源域主门禁 `SciDB val`（`866` 图、`869` 框）上，匹配双向冲突框（`Case B + Case C`）仅 `[14, 18, 21]` 个（`Case C` 均值仅 `6.3` 个），逐框门控 $\alpha_i^* \in [0, 1]$ 参数空间总上界相对 `Abl1` 仅 `+0.76±0.30%`。
   - 源域拟合的 14 维学习门控在 `val` 5 折 OOF 上倒退 `-0.51pt`，迁移至 `testB / testC` 倒退 `-1.41pt / -0.82pt`；无正则 14 类独立权重在 `testB / testC` 倒退 `-1.27pt / -1.10pt`，强经验贝叶斯收缩则退化回全局固定权重。
2. **完成第一层 5 个零样本预注册检测校准候选（`C1–C5`）三随机种子评测与 `B=1,000` 配对 Bootstrap**：
   - 全部 5 个候选在 `SciDB testB / testC` 及外部果园 `Zenodo 11148332`（`519` 图）上均实现 **3/3 随机种子稳定超越 `Native YOLOv8n` 与 `Full DINOv2`**（配对 Bootstrap 双侧 $p \le 0.004$）：
     - `C1_Fast1Crop_StaticCalib`（`1.0×` ViT）：`testB 78.30±1.94%`、`testC 83.83±1.38%`；
     - `C2_Fast1Crop_OOD_CertaintyPool`（`1.0×` ViT，马氏流形衰减）：`testB 78.07±1.68%`、`testC 83.96±1.52%`；在室内近景单叶 `CitrusUAT`（`500` 图）上自动识别检测器流形偏移（均值 `z_Maha = +0.63`），3/3 种子较静态 `C1` 追回 `+0.81 ~ +1.82pt` Macro-F1（同时完整披露纯单叶无定位场景下 `Full DINOv2` `73.44%` 仍高于融合模型 `70.24%` 的边界事实）；
     - **`C3_ZeroMLP_DualCrop_CoreStd`（`2.0×` ViT，零多尺度 MLP 参数，`0.82× Core + 1.25× Std` 纯冻结 `LP-FT` 对数池化）**：取得 **`testB 79.19±1.10%`、`testC 84.96±1.38%`**，统计显著优于单框 `C1`（`+0.91pt`, $p=0.014$ / `+1.13pt`, $p<0.001$），与三尺度 MLP 集成 `C5a/C5b`（`79.17–79.27% / 84.81–84.94%`）完全持平（$p > 0.50$），省去 `33%` ViT 算力并消除跨柑橘品种 MLP 过拟合风险；
     - `C4_HardBudget25_MultiScale`（`1.5×` ViT 硬封顶，严格仅升级前 `25%` 高分歧框）：`testB 79.03±1.40%`、`testC 84.55±1.25%`；
     - `C5a / C5b`（`3.0×` ViT 云端三尺度集成）：`testB` 交互式 `Top-1 / Top-2 / Top-3` 候选命中率达 `93.1% / 98.0% / 99.2%`，传染性病害 vs 非生物缺素胁迫 OOD 后验区分 `ROC-AUC = 0.9997`。
3. **完成第二层保形风险控制（`R2`）与视觉/PCR 双场景分层送审（`R1 & R3`）200 次互斥划分验证**：
   - **`R2` 有限样本保形风险控制（CRC）**：在 $n_{\text{cal}}=80$、严苛风险上限 $\alpha=10\%$ 下，`C1–C5` 在 `testB` 上实测留出集风险为 `9.19%–9.54%`（严格满足 $\le 10\%$，而无有限样本校正的朴素阈值在 `B0` 上超限达 `11.29%`），同时将自动放行覆盖率从 `B0` 的 `79.98%` 大幅提升至 **`93.32%–94.73%`**。
   - **`R1` vs `R3` 双任务严格区分**：零标签校准后验不确定度 `R1a (1 - max_p)` 在 `testB` 视觉外观错判捕获上达 `35.3% (5%) / 60.1% (10%) / 88.7% (20%)`；而在 `597` 例 PCR 疑似症状对齐集上（严守“黄化不等于确诊黄龙病”），单纯最小置信度在 `5%` 预算仅捕获 `6.56%`（负向对照），我们提出的 **`R3` 软后验经验贝叶斯期望风险**（$n_{\text{cal}}=100$）在 `G1` 下达 **`19.97% / 37.59% / 66.77%`**（`G2` 下达 `19.78% / 37.48% / 65.41%`），全预算档位稳定超越 DS 离散硬类别先验（`G1`: `17.03% / 32.67% / 56.24%`）。
4. **交付真实果园即插即用 CLI 工具**：编写并通过自检 `orchard_plug_and_play_cli.py`，支持未标注果园图 `Top-3` 候选卡片预标注、缺素胁迫提示与保形阈值计算。

## 证据与来源
- 前置只读可行性审计：`antigravity/v4_orchard_candidate_arsenal/scripts/phase0_readonly_oracle_and_predictability_audit.py`、`results/phase0_readonly_oracle_and_predictability_results.json`
- 两层八候选完整评测套件：`antigravity/v4_orchard_candidate_arsenal/scripts/run_candidate_arsenal_and_conformal_suite.py`、`results/orchard_candidate_arsenal_complete.json`
- 即插即用真实果园 CLI：`antigravity/v4_orchard_candidate_arsenal/scripts/orchard_plug_and_play_cli.py`
- 选型指南与全景图表：`antigravity/v4_orchard_candidate_arsenal/docs/01_面向真实果园落地的两层八候选预注册方法库与保形风险控制指南.md`、`figures/fig1_layer1_candidate_arsenal_and_ood_guard.png`、`figures/fig2_layer2_conformal_risk_and_pcr_triage.png`

## 修改文件
- `antigravity/README.md`
- `antigravity/v4_orchard_candidate_arsenal/docs/01_面向真实果园落地的两层八候选预注册方法库与保形风险控制指南.md`
- `antigravity/v4_orchard_candidate_arsenal/scripts/phase0_readonly_oracle_and_predictability_audit.py`
- `antigravity/v4_orchard_candidate_arsenal/scripts/run_candidate_arsenal_and_conformal_suite.py`
- `antigravity/v4_orchard_candidate_arsenal/scripts/orchard_plug_and_play_cli.py`
- `antigravity/v4_orchard_candidate_arsenal/scripts/generate_candidate_arsenal_figures.py`
- `antigravity/v4_orchard_candidate_arsenal/results/phase0_readonly_oracle_and_predictability_results.json`
- `antigravity/v4_orchard_candidate_arsenal/results/orchard_candidate_arsenal_complete.json`
- `antigravity/v4_orchard_candidate_arsenal/figures/fig1_layer1_candidate_arsenal_and_ood_guard.png`
- `antigravity/v4_orchard_candidate_arsenal/figures/fig2_layer2_conformal_risk_and_pcr_triage.png`
- `reports/2026-11-26/antigravity-orchard-candidate-arsenal-and-conformal-suite.md`

## 验证命令与结果
- 环境：`Windows 11` + `D:\anaconda\envs\minidector\python.exe`（`PyTorch 2.5.1+cu121`, 严格 `FP32`, `TF32=False`）
- 保形风险控制 CLI 自检：`python antigravity/v4_orchard_candidate_arsenal/scripts/orchard_plug_and_play_cli.py --mode crc_demo` -> `[CRC 自检通过] α=0.15 -> λ_CRC=0.2840`（退出码 `0`）
- 三随机种子完整候选库评测：`python antigravity/v4_orchard_candidate_arsenal/scripts/run_candidate_arsenal_and_conformal_suite.py` -> 6 份冻结权重 `SHA-256` 100% 吻合，耗时 `242.93s` 完整输出 JSON（退出码 `0`）
- 全景对比图生成与视觉核验：`python antigravity/v4_orchard_candidate_arsenal/scripts/generate_candidate_arsenal_figures.py` -> 生成 2 张无遮挡高清 PNG（退出码 `0`）
- 仓库合规校验：`python scripts/validate_repository.py --changed-files <tmp>` -> `仓库校验通过。`

## 未完成或阻塞
- 常山胡柚 / 香柚真实果园图像的专家复核真值与合作方授权照片尚未回填（0 份已完成盲评）；`SciDB testC` 中 `S12` 原始标签映射仍待核查原始 `label_object.json`，故本轮所有 `testB / testC / Zenodo` 数字严格定位为预冻结方法库的跨条件/跨数据集对比证据，最终目标域定案留待真实胡柚盲测集一次性运行裁决。

## 对下游的影响
- 为真实胡柚果园数据到位后的各阶段（0 标签预标注、端侧单框实时推理、跨品种双尺度零 MLP 推理、云端三尺度高精推理、40~100 样本保形风险门禁及 PCR 疑似症状优先送检）提供了预冻结、零泄漏、一行命令可调用的完整候选方案矩阵，彻底消除临时拼凑模块导致的数据泄漏风险。

## 下一步建议
- 冻结当前 `v4` 两层八候选方法库，不再在历史公开数据集上迭代模型结构；优先用 `orchard_plug_and_play_cli.py --mode preannotate --candidate C3` 辅助整理胡柚果园候选框供专家高效复核，并在获得约 80~100 个采样隔离簇真值后按预注册流程一次性完成盲测选型与保形阈值锁定。
