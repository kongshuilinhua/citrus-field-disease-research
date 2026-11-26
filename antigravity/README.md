# Antigravity 独立研究、方法学审计与真实果园预注册候选方法库归档（与 GPT、DeepSeek 成果严格隔离）

> **隔离归档说明**：本目录（`antigravity/`）专门存放 **Antigravity** 独立完成的可行性审计、预注册候选方法库、保形风险控制实现、统计显著性检验与方法学边界报告，与仓库内 **GPT / 协调者**（`coordination/`、`docs/项目现状_*.md`、`docs/research/`）及 **DeepSeek（DS）**（`docs/experiments/`）完全分目录独立归档，互不混写、互不覆盖。

---

## 一、最新核心成果：面向真实果园落地的两层八候选预注册方法库与保形风险控制套件（2026-11-26 `v4`）

- **完整技术与选型指南**：**[v4_orchard_candidate_arsenal/docs/01_面向真实果园落地的两层八候选预注册方法库与保形风险控制指南.md](v4_orchard_candidate_arsenal/docs/01_面向真实果园落地的两层八候选预注册方法库与保形风险控制指南.md)**
- **全景对比配图（含全部基线与负向对照）**：
  - [fig1_layer1_candidate_arsenal_and_ood_guard.png](v4_orchard_candidate_arsenal/figures/fig1_layer1_candidate_arsenal_and_ood_guard.png)（第一层 5 个零样本检测候选 `C1–C5` 三种子精度梯度、外部果园 `Zenodo` 与室内近景单叶 `CitrusUAT` OOD 防护验证）
  - [fig2_layer2_conformal_risk_and_pcr_triage.png](v4_orchard_candidate_arsenal/figures/fig2_layer2_conformal_risk_and_pcr_triage.png)（第二层 `R2` 有限样本保形风险控制自动放行覆盖率 & `R1` 视觉纠错 vs `R3` 软后验经验贝叶斯 `597` 例 PCR 疑似症状分层送检）
- **核心亮点摘要（100% 冻结权重复用、零测试集调参、不作假）**：
  1. **前置只读可行性审计提前阻断死胡同**：定量证明源域 `SciDB val` 双向匹配冲突仅 `[14, 18, 21]` 个（`Case C` 均值仅 `6.3` 个），逐框门控参数空间总天花板仅 `+0.76%`，源域训练的 14 维学习门控与无正则 14 类权重迁移至 `testB / testC` 分别发生 `-1.41pt / -0.82pt` 与 `-1.27pt / -1.10pt` 的负迁移，确立早停复杂学习门控的统计依据。
  2. **第一层（路线 1）构建 5 个零样本互补检测候选（`C1–C5`）全线 3/3 种子超越强基线**：
     - **`C1`（单框静态对数校准，`1.0×` ViT）**：`testB 78.30% / testC 83.83%`，边缘实时默认档；
     - **`C2`（单框 OOD 流形衰减确信池，`1.0×` ViT）**：`testB 78.07% / testC 83.96%`，在室内单叶 OOD `CitrusUAT` 上 3/3 种子较静态 `C1` 追回 `+0.81~+1.82pt` Macro-F1（同时诚实披露纯单叶摆拍下 `Full DINOv2` `73.44%` 仍最优）；
     - **`C3`（零 MLP 参数双尺度 `0.82× Core + 1.25× Std` 纯冻结池化，`2.0×` ViT）**：**无需任何多尺度 MLP 训练参数**即达 **`testB 79.19% / testC 84.96%`**（较 `B1` 配对 Bootstrap `+3.99% / +2.82%`, $p < 0.001$；较 `C1` 显著提升 `+0.91% / +1.13%`），与三尺度 MLP 集成 `C5` 统计持平但省去 `33%` ViT 算力与跨品种 MLP 过拟合风险；
     - **`C4`（25% 硬预算自适应升级，`1.5×` ViT 严格封顶）**：`testB 79.03% / testC 84.55%`，彻底解决固定阈值升级比例失控问题；
     - **`C5`（云端三尺度集成 `C5a/C5b`，`3.0×` ViT）**：`testB 79.17–79.27% / testC 84.81–84.94%`，`Top-3` 可确认症状候选命中率达 **`99.0%–99.2%`**。
  3. **第二层（路线 2）提供严格互斥划分的保形风险控制与双场景复核候选（`R1–R3`）**：
     - **`R2`（有限样本保形风险控制 CRC，$n_{\text{cal}}=80, \alpha=10\%$）**：将自动放行覆盖率从原生 `YOLOv8n` 的 `80.0%` 提升至 **`93.3%–94.7%`**（减少约 `70%` 人工复核量）且实测风险严格受控（`9.19%–9.54%`）；
     - **`R3`（软后验经验贝叶斯期望风险，`597` 例 PCR 对齐集 200 次互斥切分）**：在 `n_cal=100`、`5%/10%/20%` 预算下捕获 **`20.0% / 37.6% / 66.8%`**（`G1`），稳定超越 DS 离散硬类别先验（`17.0% / 32.7% / 56.2%`），全程严守“黄化仅作疑似筛查、不等于确诊黄龙病”规范。

---

## 二、核心必读：独立创新性与方法学严格审计报告（2026-11-26 `v3` 核定，零包装）

- **审计总报告**：**[v3_bgopta_and_dtemop/docs/02_独立创新性与方法学严格审计报告_20261126.md](v3_bgopta_and_dtemop/docs/02_独立创新性与方法学严格审计报告_20261126.md)**
- **审计图表（含全部强基线与负结果）**：
  - [fig1_complete_baselines_negatives_and_bootstrap.png](v3_bgopta_and_dtemop/figures/fig1_complete_baselines_negatives_and_bootstrap.png)
  - [fig2_fair_e2e_speed_and_external_with_negatives.png](v3_bgopta_and_dtemop/figures/fig2_fair_e2e_speed_and_external_with_negatives.png)

---

## 三、目录导航与模型权重溯源

- **`v4_orchard_candidate_arsenal/`**：前置只读 Oracle/AUC 可行性审计、两层八候选预注册方法库（`C1–C5` + `R1–R3`）、有限样本保形风险控制（CRC）与真实果园即插即用 CLI（`orchard_plug_and_play_cli.py`）。
- **`v3_bgopta_and_dtemop/`**：`BG-OPTA` 纯源域零泄漏负结果探针、同几何背景干预、逐图 Bootstrap 配对检验、含 NMS 公平端到端测速与独立创新性审计报告。
- **`v2_tcd_mge/`**：`TCD-MGE` 与 `Abl1 / Abl2 / Abl3` 三种子实验记录与脚本（已附规范纠偏说明）。
- **根级 `docs/ scripts/ results/`**：早期 `MGS-SymTex (v1)` 探索记录（负向对照基线）；本地全部冻结权重 `SHA-256` 清单见 [MODEL_CHECKPOINTS_MANIFEST.csv](MODEL_CHECKPOINTS_MANIFEST.csv)。
