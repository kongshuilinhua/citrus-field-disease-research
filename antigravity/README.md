# Antigravity 独立研究与创新性审计归档（与 GPT、DeepSeek 成果严格隔离）

> **隔离归档说明**：本目录（`antigravity/`）专门存放 **Antigravity** 独立完成的方法实现、受控消融、统计显著性检验与方法学审计，与仓库内 **GPT / 协调者**（`coordination/`、`docs/项目现状_*.md`、`docs/research/`）及 **DeepSeek（DS）**（`docs/experiments/`）完全分目录独立归档，互不混写、互不覆盖。

---

## 一、核心必读：独立创新性与方法学严格审计报告（2026-11-26 最新核定，零包装）

- **审计总报告**：**[v3_bgopta_and_dtemop/docs/02_独立创新性与方法学严格审计报告_20261126.md](v3_bgopta_and_dtemop/docs/02_独立创新性与方法学严格审计报告_20261126.md)**
- **审计图表（含全部强基线与负结果）**：
  - [fig1_complete_baselines_negatives_and_bootstrap.png](v3_bgopta_and_dtemop/figures/fig1_complete_baselines_negatives_and_bootstrap.png)（`SciDB` 全基线 + 负结果探针 + `B=1000` 逐图配对 Bootstrap 置信区间 & 同几何背景—分辨率解耦实验）
  - [fig2_fair_e2e_speed_and_external_with_negatives.png](v3_bgopta_and_dtemop/figures/fig2_fair_e2e_speed_and_external_with_negatives.png)（含完整融合重算分与 `batched_nms` 的公平端到端延迟 & 含强基线反超负结果的外部/单叶基准对照）

### 七项严格审计核心结论摘要（不改冻结权重、不新增方法名包装旧 `Abl3`）

1. **同几何背景干预（区分有效目标分辨率下降与背景串扰）**：
   - 在固定 `1.00×` 框内叶片有效像素完全不变的条件下，从 `1.25×` 扩窗至 `1.50×`（中性灰外环 `C2 → C4`）仅因**有效目标分辨率下降 `30.6%` 与尺度失配**即造成纯冻结 `DINOv2` 在 `testB / testC` 下降 **`-2.08pt / -2.02pt`**；而同几何真实外环背景（`C4 → C3`）在 `testB` 仅占 **`-0.68pt`**、在白板反光 `testC` 占 **`-2.56pt`**。
   - 单趟 `1.50×` 空间 Token 正交剥离探针 **`BG-OPTA (S3)`** 仅在异图纹理杂波外环（`C6`）下较纯 `[CLS]` 有正收益（`+6.42pt / +3.89pt`），在均匀灰/白板外环下反而负向过剥离（`-1.97pt ~ -2.94pt`），在主任务 `SciDB testC`（`79.31%` vs `Full DINOv2` `83.30%`）上属于**明确的负结果探针**。
2. **三尺度普通拼接 vs 差分表示（`Abl3`）的逐种子配对消融与 `B=1,000` 逐图 Bootstrap**：
   - 废止用新缩写（如 `IGD-DCR`）重新包装旧 `Abl3` 结果的做法，一律使用原始消融名称 `Abl1 / Abl2 / Abl3`。
   - **显式目标—环境差分（`Abl3`: `79.28% / 85.09%`）相对普通三尺度特征拼接 MLP（`79.24% / 84.92%`）无统计显著性**：`testB` 配对均值差 **`+0.049%`（95% CI `[-0.211%, +0.300%]`, $p = 0.704$；Seed 0 为 `-0.17%`）**，`testC` 配对均值差 **`+0.138%`（95% CI `[-0.229%, +0.458%]`, $p = 0.430$）**。
   - 真正具有统计显著性的增益来自两步：**① 单框 `Abl1`（锚点 14 类对数线性融合 `+2.56pt` + 统一几何分数重加权 `+0.30pt` + 重判后同类别 NMS `+0.32pt`）** 在 `testB / testC` 共提升 `+3.18pt / +1.87pt`；**② 引入三尺度裁剪集成（`Abl3 - Abl1`）** 在三种子均值上进一步提升 **`+1.069%`（`p=0.014`）/ `+1.071%`（`p=0.018`）**。
3. **纠正 `YOLOv8` 检测头分数的术语误用**：
   - 源码与实证确认 `YOLOv8n` `Detect` 头为 `64 (DFL) + 14 (cls) = 78` 通道，**无独立 Objectness 分支**；在 `IoU ≥ 0.90` 的高定位精度框上，分类错误框的 `max_c sigmoid` 置信度较分类正确框暴跌 **`-0.159 ~ -0.182`**。已全面纠正为**“检测器最高类别 Sigmoid 置信度与二阶段后验的统一几何重加权”**。
4. **含完整分类头、后验融合、重算分与 `batched_nms` 的公平端到端测速（RTX 4060 Ti, 严格 FP32）**：
   - 实测没有任何多视角方法快于单框 `Full DINOv2`（`val 15.46 ms / testC 14.64 ms`）：`Abl1` 为 `16.79 / 15.82 ms`，`Abl3` 为 **`val 22.85 ms (多叶图 P95=46.66 ms) / testC 18.86 ms`**，`Abl2` 与 `TCD-MGE (v2)` 为 `26.79–27.53 ms / 25.20–25.79 ms`。
5. **全数据集历史角色规范化、伪掩膜披露与外部负结果补全**：
   - `SciDB testB/testC`（同对象配对历史探索集）、`Zenodo 11148332`（多轮复用探索性外部集）、`Sweet Orange V1`（**曾用于 `v1` 分类验证的单叶分类集，无框标注，非零接触检测盲测**；且其 3 类子集 `Macro-F1` 上原生 `YOLOv8n` `80.24%` 高于 `Abl3` `73.22%` 与 `Full DINOv2` `61.27%`）；
   - `CitrusUAT`（历史受控深灰底单叶集：其中叶片掩膜系**四边灰底色差阈值生成的启发式伪掩膜**而非人工真值分割；且在 5 类跨域 `Macro-F1` 上**单纯 `Full DINOv2` `73.44%` 反超 `Abl3` `69.88%`、`TCD-MGE` `69.39%` 与 `BG-OPTA` `64.36%`**）。

---

## 二、目录历史留档导航

- **`v3_bgopta_and_dtemop/`**：`BG-OPTA` 纯源域零泄漏探针实现、修复版 `Zenodo` 评测、完整四项定量审计脚本（`run_rigorous_audit_suite.py`）、`rigorous_audit_suite_complete.json` 及独立创新性审计报告。
- **`v2_tcd_mge/`**：`TCD-MGE` 与 `Abl1 / Abl2 / Abl3` 三种子实验记录与脚本（已附规范纠偏说明）。
- **根级 `docs/ scripts/ results/`**：早期 `MGS-SymTex (v1)` 探索记录（已核定为负向对照基线 `B2`）；本地全部冻结权重 `SHA-256` 见 [MODEL_CHECKPOINTS_MANIFEST.csv](MODEL_CHECKPOINTS_MANIFEST.csv)。
