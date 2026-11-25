# Antigravity 独立研究归档：MGS-SymTex 跨场景柑橘病征识别小论文完整实验

> **隔离存放说明**：按仓库协作规范与管理要求，本目录（`antigravity/`）专门存放 **Antigravity** 独立完成的三路线创新筛选与 SCI 小论文全套收官实验成果，与仓库内 **GPT / 协调者**（`coordination/`、`docs/项目现状_*.md`、`docs/research/`）及 **DeepSeek（DS）**（`docs/experiments/` 下 `*_20261123.md`、本地 `DS执行_*` / `DeepSeek分析_*`）的文档和产物**完全分目录独立归档，互不覆盖**。

---

## 一、核心成果速览（全 3 随机种子 `0, 1, 2`、严格源域 `val` 选参、四源外部封口盲测）

针对柑橘跨背景检测中的**单阶段高置信误判**与**朴素全量两阶段上下文剥离退化（耗时翻倍且 `testB` 掉点 `-4.63pt`）**两大痛点，提出 **MGS-SymTex** 框架：

1. **颈部多尺度流形马氏选择性级联（`MGS`）**：钩取 `YOLOv8n` 检测头分类分支倒数第二层特征（`Detect.cv3[s][1]`，192 维），仅按源域 `SciDB val` 标定 `q=0.80` 冻结阈值（源域仅触发 **20.0%** 高分框），无需任何目标域样本即可在 `testB` / `testC` / 外部果园自动将触发率抬升至 **`48.2%` / `44.9%` / `77.9%`**：
   - **端侧零增耗轻量档（`3.35M` 剪枝）**：无教师 `MGS + SymTex-Pruned` 将 `testC AP50-95` 从原生 `72.77±1.32%` 提升至 **`77.13±2.06%`（`+4.36pt`）**，蒸馏版 `MGS + Vanilla-KD` 达 **`77.30±1.73%`（`+4.53pt`）**；
   - **高精度旗舰档（`Ours-Pro: MGS + DINOv2`）**：`val AP50-95` 达 **`90.12±0.60%`**（超越原生 `89.78%` 与全量级联 `89.46%`，单图延迟降至 **`12.10 ms` ≈ 82.6 FPS**），`testB / testC` 分别达 **`74.59±2.60%`（`+3.34pt`）** 与 **`81.68±1.36%`（`+8.91pt`，较同分类器低置信度门控高 `+1.32pt`）**。
2. **零推理增耗的病理双先验特征塑形（`SymTex`）**：训练期引入**双侧非对称空间差分（`BASD`）**与**局部亮度归一化多尺度高频残差对齐（`II-HFR`）**，推理期一键剪除全部辅助分支，严格保持 **`3.347M`** 零额外参数：
   - 在 `CitrusUAT` 上叶片掩膜内特征能量占比达 `68.53±0.25%`（`+BASD` 单项达 `70.00±0.44%`，较局部全局拼接 `B4` 的 `54.31%` 高出 `+14.22pt`）；
   - 在完全未见的**外部真实果园 `Zenodo 11148332`（519 图 / 1,765 个叶框，健康叶 vs 斑驳黄化疑似 vs 缺素对称黄化疑似 3 类细粒度病理拓扑）**一次性零样本盲测中，宏 F1 呈清晰阶梯跃升：单流基线 `56.67±4.30%` → 纯蒸馏 `Vanilla_KD` `58.71±1.58%` → 无教师 `SymTex_Pruned` **`59.05±3.03%`** → **`Ours_SymTex_KD_Pruned` `60.82±2.04%`（`+4.15pt` 胜基线、`+2.11pt` 胜纯蒸馏）**；
   - 在 `Sweet Orange V1`（`N=1,135`）零样本盲测中，`Ours_SymTex_KD_Pruned` 宏 F1 达 **`77.07±4.66%`（`+4.43pt` 胜基线）**；在跨树种柚叶 `Pomelo V2`（`N=292`）上，`+BASD` 将宏 F1 从 `63.33%` 拉升至 **`68.67±1.74%`（`+5.34pt`）**。

---

## 二、目录结构与文件索引

### 1. 论文定稿报告与预注册实验记录（`docs/`）
- **[00_SCI小论文完整实验结果与写作定稿.md](docs/00_SCI小论文完整实验结果与写作定稿.md)**：推荐题目、方法公式、表 1～表 3 完整 3 种子结果、配图图注及 Discussion 边界。
- [01_总体设计与三路线筛选协议.md](docs/01_总体设计与三路线筛选协议.md)：漏斗筛选协议与预注册晋级/淘汰门。
- [02_三路线实测裁决与实施方案.md](docs/02_三路线实测裁决与实施方案.md)：`Probe-A1 / A2 / B0 / C0` 筛选实测裁决（含阴性路线 `Probe-B0` 与 `Probe-A2` 如实归档）。
- [实验记录_小论文完整实验_20261125.md](docs/实验记录_小论文完整实验_20261125.md)：实验 E1～E4 完整运行记录。
- [实验记录_三路线创新筛选_20261124.md](docs/实验记录_三路线创新筛选_20261124.md)：前期探针实验完整登记与复盘。

### 2. 300 DPI 出版级论文插图（`figures/`）
- [fig1_pareto_ap_latency.png](figures/fig1_pareto_ap_latency.png)：跨场景精度-单图延迟帕累托前沿 + 零监督自适应触发率。
- [fig2_symtex_ablation_and_external.png](figures/fig2_symtex_ablation_and_external.png)：`3.35M` 零额外参数 3 种子组件消融 + 外部公开集零样本泛化与叶片掩膜聚焦度。
- [fig3_external_orchard_blind_det.png](figures/fig3_external_orchard_blind_det.png)：外部真实果园 519 图 / 1,765 框零样本检测与 3 类病理拓扑框分类。

### 3. 可复现实验脚本（`scripts/`）
- **小论文收官实验套件（`scripts/paper_suite/`）**：
  - `exp1_symtex_ablation_and_kd.py`：6 组组件消融与蒸馏配置 × 3 随机种子训练及 `3.347M` 结构剪枝导出。
  - `exp2_end2end_mgs_cascade_and_timing.py`：`路由策略 × 分类表征` 正交矩阵 3 种子严格 FP32 `COCO` 评测及 RTX 4060 Ti 实测单图延迟。
  - `exp3_external_blind_validation.py`：冻结权重与源域阈值下的四源独立公开集一次性零样本封口盲测。
  - `exp4_generate_paper_figures.py`：生成 300 DPI 零遮挡论文配图。
- **前期三路线探针脚本（`scripts/screening/`）**：
  - `probe_a1_selective_routing.py`、`probe_a2_hierarchical_edl.py`、`probe_a2_domain_debiased_osr.py`、`probe_b0_context_ortho_det.py`、`probe_c0_pathology_dual_stream.py`。

### 4. 原始结果摘要与模型权重校验清单（`results/` & `MODEL_CHECKPOINTS_MANIFEST.csv`）
- `results/e1_symtex_ablation_and_kd_summary.json`
- `results/e2_end2end_mgs_symtex_summary.json`
- `results/e3_external_blind_validation_summary.json`
- `results/probe_*.json`（含阳性与阴性探针完整结果）
- **[MODEL_CHECKPOINTS_MANIFEST.csv](MODEL_CHECKPOINTS_MANIFEST.csv)**：按仓库大文件策略，19 个本地 `.pt` 模型权重不入库，本表完整记录其本机路径、字节数与 `SHA-256` 校验值。
