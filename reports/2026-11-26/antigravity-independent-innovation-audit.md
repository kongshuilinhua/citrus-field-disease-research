# Antigravity 独立创新性与方法学严格审计报告（零新模型、零改名包装、零权重改动）

## 任务
在不新增模型架构、不新增方法缩写包装旧 `Abl3` 结果、不改动任何冻结权重的前提下，逐项完成七项严格方法学与统计显著性审计：
1. 完成同几何背景干预实验，严格区分背景信息串扰与 `1.25× → 1.50×` 有效目标分辨率下降；
2. 完成三尺度普通拼接与显式差分表示（`Abl3`）的逐种子受控配对消融，采用 `B=1,000` 逐图配对 Bootstrap 给出 AP 差值、95% 置信区间与双侧 $p$ 值；
3. 审核 `YOLOv8` 检测头源码与实测输出，纠正将最高类别 Sigmoid 分数称为独立 Objectness 或纯定位质量的表述，并逐项拆解 `Abl1` 增益；
4. 完成包含最终分类头、后验融合、分数重算与 `batched_nms` 的公平端到端测速（并在全量 602 张 `testC` 上实时验算输出 AP）；
5. 补全所有图表中的强基线反超现象与负结果，明确披露 `CitrusUAT` 灰底色差启发式伪掩膜属性；
6. 全面核定 `SciDB testB/testC`、`Zenodo 11148332`、`Sweet Orange V1`、`CitrusUAT` 的历史使用角色；
7. 输出客观的独立创新性审计报告，逐项说明增益来源、机制证据、局限性及是否具备投稿价值。

## 实际完成
1. **完成同几何背景干预实验（分离分辨率损失与背景影响）**：
   - 在固定 `1.00×` 框内叶片有效像素完全恒等、零 `YOLOv8` 融合（`alpha_det=0`）的纯视觉评估下：`1.25× 灰外环 → 1.50× 灰外环`（纯有效目标分辨率下降 `30.6%`）单独导致冻结 `DINOv2` 在 `testB / testC` 下降 **`-2.08pt / -2.02pt`**；同几何 `1.50× 灰外环 → 1.50× 实景外环` 在 `testB` 占 **`-0.68pt`**、在白板反光 `testC` 占 **`-2.56pt`**。
   - `BG-OPTA (S3)` 仅在异图纹理杂波外环（`C6`）下较纯 `[CLS]` 提升 `+6.42pt / +3.89pt`，而在均匀灰/白板外环下因误剥离低频亮度分量反而下降 `-1.97pt ~ -2.94pt`，主任务 `testC`（`79.31%` vs `Full DINOv2` `83.30%`）确认为负结果。
2. **完成三尺度普通拼接 vs 差分表示（`Abl3`）逐种子配对消融与 `B=1,000` 逐图 Bootstrap**：
   - `Abl3`（`79.28% / 85.09%`）减三向量普通拼接 `PlainConcat_3Vec`（`79.24% / 84.92%`）：`testB` 均值差 **`+0.049%`（95% CI `[-0.211%, +0.300%]`, $p=0.704$，Seed 0 为 `-0.17%`）**，`testC` 均值差 **`+0.138%`（95% CI `[-0.229%, +0.458%]`, $p=0.430$）**——**显式差分相对普通拼接无统计显著性**。
   - 统计显著的真实增益来自两层：① 单框 `Abl1`（14 类对数融合 `+2.56pt / +0.82pt` + 统一几何分数重加权 `+0.30pt / +0.55pt` + 重判后同类别 NMS `+0.32pt / +0.50pt`）共提升 `+3.18pt / +1.87pt`；② 三尺度裁剪集成（`Abl3 - Abl1`）三种子均值提升 **`+1.069%`（`p=0.014`）/ `+1.071%`（`p=0.018`）**。
3. **完成 `YOLOv8` 检测头审核与术语纠偏**：
   - 核验 `Detect` 头通道为 `64 (DFL) + 14 (cls) = 78`（无独立 Objectness 分支）；在 `IoU ≥ 0.90` 匹配框上，分类错误框的 `max_c sigmoid` 较分类正确框低 **`-0.159 ~ -0.182`**，已在全部文档中纠正错误术语并废止 `IGD-DCR` 新名包装。
4. **完成含完整融合、重算分与 `batched_nms` 的端到端公平测速（全集 602 张 `testC` 实时验算 AP 一致）**：
   - RTX 4060 Ti 严格 FP32 下：原生 `YOLOv8n` 为 `8.10 / 8.06 ms`，单框 `Full DINOv2` 为 **`15.46 / 14.64 ms`**，`Abl1` 为 `16.79 / 15.82 ms`，`Abl3` 为 **`val 22.85 ms (P95 46.66 ms) / testC 18.86 ms`**，`Abl2` 与 `TCD-MGE (v2)` 为 `26.79–27.53 ms / 25.20–25.79 ms`（无任何多视角方法快于单框 `Full DINOv2`）。
5. **补全全部负结果图表、明确伪掩膜属性与历史角色，输出独立创新性审计报告**：
   - 明确披露 `CitrusUAT` 掩膜系四边灰底色差阈值生成的**启发式伪掩膜**，且 `CitrusUAT` 5 类 `Macro-F1` 上单框 `Full DINOv2`（`73.44%`）反超所有融合模型（`Abl3` `69.88%`、`BG-OPTA` `64.36%`）；明确 `Sweet Orange V1` 曾用于早期 `v1` 分类实验且无框真值，其 3 类子集 `Macro-F1` 上原生 `YOLOv8n`（`80.24%`）高于所有二阶段模型。

## 证据与来源
- 四合一严格审计脚本：`antigravity/v3_bgopta_and_dtemop/scripts/run_rigorous_audit_suite.py`
- 完整审计与 Bootstrap JSON：`antigravity/v3_bgopta_and_dtemop/results/rigorous_audit_suite_complete.json`
- 独立创新性审计报告：`antigravity/v3_bgopta_and_dtemop/docs/02_独立创新性与方法学严格审计报告_20261126.md`
- 含全部强基线与负结果的审计配图：`antigravity/v3_bgopta_and_dtemop/figures/fig1_complete_baselines_negatives_and_bootstrap.png`、`fig2_fair_e2e_speed_and_external_with_negatives.png`

## 修改文件
- `antigravity/README.md`
- `antigravity/v2_tcd_mge/docs/01_TCDMGE_SCI小论文定稿与3比3全胜验证报告.md`
- `antigravity/v3_bgopta_and_dtemop/docs/01_SCI小论文终稿立论_输入隔离差分与输出Token正交探针完整报告.md`
- `antigravity/v3_bgopta_and_dtemop/docs/02_独立创新性与方法学严格审计报告_20261126.md`
- `antigravity/v3_bgopta_and_dtemop/scripts/run_rigorous_audit_suite.py`
- `antigravity/v3_bgopta_and_dtemop/scripts/generate_honest_audit_figures.py`
- `antigravity/v3_bgopta_and_dtemop/results/rigorous_audit_suite_complete.json`
- `antigravity/v3_bgopta_and_dtemop/figures/fig1_complete_baselines_negatives_and_bootstrap.png`
- `antigravity/v3_bgopta_and_dtemop/figures/fig2_fair_e2e_speed_and_external_with_negatives.png`
- `reports/2026-11-26/antigravity-independent-innovation-audit.md`

## 验证命令与结果
- 冻结权重零改动核验与四项定量审计：`python run_rigorous_audit_suite.py` -> 退出码 `0`，6 份冻结权重 `SHA-256` 100% 吻合，精确 Bootstrap 与 `COCOeval_faster` 误差 `<1e-5%`，`testC` 602 图实时测速输出框验算 Seed 0 `AP50-95` 与离线结果一致（`71.33% / 85.43% / 86.36%`）。
- 审计配图生成：`python generate_honest_audit_figures.py` -> 退出码 `0`，视觉核验无遮挡。
- 仓库合规校验：`python scripts/validate_repository.py --changed-files <tmp>` -> `仓库校验通过。`

## 未完成或阻塞
- 当前所有公开柑橘数据均存在历史接触记录或缺乏多目标复杂田间边界框标注；真正独立的最终验证仍需等待自采常山胡柚图像完成专家盲评回填或新采配对田间数据。

## 对下游的影响
- 彻底清除了过度包装、术语失准与统计不显著模块的夸大宣传，明确了真实有效增益边界（`Abl1` 14 类后验融合+统一分数重加权+重判后 NMS 与 `Abl3` 多尺度裁剪集成）及跨域负迁移风险（室内纯灰底单叶上弱检测器融合的拖累）。

## 下一步建议
- 停止继续在现有历史复用集（`SciDB testB/testC`、`Zenodo`）上叠加新模块或调参；若撰写硕士应用型小论文，应诚实围绕两阶段级联失效诊断、锚点全类后验校准融合（`Abl1`）与多尺度裁剪集成（`Abl3`）展开，完整报告配对 Bootstrap 置信区间与负结果，并优先推进真实田间独立复核真值的建设。
