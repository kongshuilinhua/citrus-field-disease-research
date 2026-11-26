# Antigravity 终稿闭环实验报告：零泄漏输出层空间 Token 正交探针（BG-OPTA）与输入隔离多粒度差分精化（IGD-DCR）

## 任务
围绕跨果园背景柑橘病害检测 SCI 小论文定稿需求，全面响应并逐条解决前期 `TCD-MGE (v2)` 复核中发现的四项关键问题：
1. 在**物理零接触任何目标测试集**的前提下，预注册、实现并评测单趟 `1.50×` 框引导空间 Token 背景子空间正交剥离与病斑峰值聚合探针（`BG-OPTA`），与同算力 `[CLS]`（`S0`）、普通空间 Token 拼接（`S1`）、仅正交拓扑（`S2`）及输入级隔离多粒度差分（`IGD-DCR`）进行严格 3 随机种子消融对比，从机理上回答“为何不能仅靠普通拼接或单框输出层背景剥离”；
2. 消除测速脚本中的重复图像缩放冗余，完成 RTX 4060 Ti 严格 FP32 **七档公平单图推理延迟与调用次数实测**，解决前期串行升级比静态融合更慢且无精度优势的矛盾；
3. 重新校准 `SciDB testB/testC` 与 `Zenodo 11148332` 为跨条件诊断/探索基准，彻底修复 `Zenodo` 3 类评测遗漏 `9–13` 类预测框的代码缺陷（实现 100% 预测框零丢弃），并在 `CitrusUAT`（`N=443`）真实物理叶片掩膜上完成无掩膜前景聚焦纯度定量核验；
4. 在从未接触过任何检测架构开发的独立公开去重原图集 **`Sweet Orange V1`（`N=1,389`）** 上执行唯一一次冻结权重封口盲测，并将全部成果严格隔离归档至 `antigravity/` 同步上传。

## 实际完成
1. **完成纯源域零泄漏阶段一训练与权重锁定（`phase1_source_only_train_and_select.py`）**：
   - 脚本物理隔离 `testB / testC / Zenodo / CitrusUAT / SweetOrange`，仅在 `SciDB train`（85% 拟合 / 15% 源内开发）与 `SciDB val` 上完成 `S0 / S1 / S2 / S3 (BG-OPTA)` 3 随机种子训练，一次性锁定 `bgopta_frozen_seed{0,1,2}.pt`（登记 SHA-256）后再进入阶段二单次评测。
   - 纯源域合成背景侵入集 `source_dev_bgshift` 上，`Macro-F1` 从 `S0 ([CLS])` 的 `83.72±0.95%`、`S1 (普通拼接)` 的 `87.44±0.67%` 提升至 `S2 (正交拓扑)` 的 `89.79±0.67%` 与 **`S3 (BG-OPTA)` 的 `90.11±0.17%`（3/3 种子全胜）**；`SciDB val AP50-95` 达 **`90.42±0.68%`**。
2. **完成完全冻结阶段二 & 二B 全基准评测，确立“ViT 自注意力前向污染”核心物理结论与精简终稿架构 `IGD-DCR`**：
   - **正交投影与真实掩膜聚焦验证**：在 `CitrusUAT`（`N=443`）真实物理叶片掩膜核验中，零掩膜输入的 `BG-OPTA` 自动抑制了框内 **`67.71%`** 的灰底四角权重，使叶肉特征纯度从 `95.84%` 升至 **`97.39%`**；在修复版零丢框 `Zenodo` 上 2类 `AP50` 达 `43.38±1.28%`（3/3 种子胜 `Full DINOv2` `41.81%` 与 `S1` `42.98%`）。
   - **触发预注册门禁并揭示 ViT 前向污染定律**：在强白板反光偏移 `SciDB testC` 上，单框扩至 `1.50×` 使外周反光补丁在 `DINOv2` 的 12 层全局自注意力内部提前污染叶片 Token，导致 `S0` 从 `1.25×` 校准版的 `83.42%` 暴跌至 `78.30%`，输出层事后正交剥离（`79.31%`）无法逆转该层内非线性串扰；而**输入级空间隔离差分 `IGD-DCR-Batch`** 取得 **`testB = 79.28±1.02%`、`testC = 85.09±1.09%`**（高出输出层剥离 `+3.34pt / +5.78pt`，9/9 项全胜 `Full DINOv2`）。
   - **七档公平测速解决速度矛盾**：查明小批量下 ViT 受限于 22M 权重显存带宽读取——**单批并行三视角 `IGD-DCR-Batch` 在 RTX 4060 Ti 上 `testC` 仅需 `18.61 ms`（`53.7 FPS`，`val` 为 `19.89 ms / 50.3 FPS`）**，比两步串行升级 `TCD-MGE v2`（`24.13 ms`）快 **`22.9%`**，仅比单框 `Full DINOv2`（`14.09 ms`）多 `4.52 ms`；同时无阈值参数的**纯分歧升级版 `IGD-DCR-Esc`** 在 `val / testB / testC` 仅升级 `3.7% / 22.9% / 26.1%` 候选框，为边缘串行处理器节省 `74%–96%` 额外 ViT FLOPs 且保持 `90.52% / 79.09% / 84.80%`。
3. **修复 `Zenodo` 14 类零丢框映射并完成 `Sweet Orange V1`（`N=1,389`）唯一一次封口盲测**：
   - 修复 `Zenodo` `0..13` 全类别完备映射（保留 100% 预测框）后，`IGD-DCR-Batch` 取得 2类 `AP50 = 45.08±2.15%`、3类 `AP50 = 30.18±1.22%`（较 `Full DINOv2` `+3.27pt / +3.26pt`，3/3 种子全胜）。
   - 在从未接触过的 `Sweet Orange V1`（`1,389` 张去重原图）单次封口盲测中，开放 14 类零遮蔽精确命中率：`Native YOLOv8n = 54.91%`、`Full DINOv2 = 56.47%`、**`IGD-DCR-Esc = 60.93%`（3/3 胜）**、**`IGD-DCR-Batch = 61.17%`（`+4.70pt`，3/3 胜）**；3 症候群 `Macro-F1` 较 `Full DINOv2`（`61.27%`）提升 **`+11.95pt`** 至 **`73.22%`**。
4. **历史文档规范纠偏与隔离归档**：在 `antigravity/v2_tcd_mge/` 补入数据集定位与速度复盘声明，新建 `antigravity/v3_bgopta_and_dtemop/` 完整归档脚本、JSON 与 300 DPI 无重叠配图。

## 证据与来源
- 纯源域锁定记录：`antigravity/v3_bgopta_and_dtemop/results/phase1_source_only_ablation_and_lock.json`
- 冻结全基准单次评测：`antigravity/v3_bgopta_and_dtemop/results/phase2_frozen_oneshot_complete_eval.json`
- 七档零冗余测速与多粒度同口径外部/盲测汇总：`antigravity/v3_bgopta_and_dtemop/results/phase2b_frozen_multigranular_and_clean_latency.json`
- 本地冻结权重 SHA-256 清单：`antigravity/MODEL_CHECKPOINTS_MANIFEST.csv`

## 修改文件
- `antigravity/README.md`
- `antigravity/MODEL_CHECKPOINTS_MANIFEST.csv`
- `antigravity/v2_tcd_mge/docs/00_预注册协议与方法设计.md`
- `antigravity/v2_tcd_mge/docs/01_TCDMGE_SCI小论文定稿与3比3全胜验证报告.md`
- `antigravity/v2_tcd_mge/docs/实验记录.md`
- `antigravity/v3_bgopta_and_dtemop/docs/01_SCI小论文终稿立论_输入隔离差分与输出Token正交探针完整报告.md`
- `antigravity/v3_bgopta_and_dtemop/docs/实验记录.md`
- `antigravity/v3_bgopta_and_dtemop/scripts/phase1_source_only_train_and_select.py`
- `antigravity/v3_bgopta_and_dtemop/scripts/phase2_frozen_oneshot_eval.py`
- `antigravity/v3_bgopta_and_dtemop/scripts/phase2b_frozen_multigranular_and_clean_latency.py`
- `antigravity/v3_bgopta_and_dtemop/scripts/generate_bgopta_figures.py`
- `antigravity/v3_bgopta_and_dtemop/results/phase1_source_only_ablation_and_lock.json`
- `antigravity/v3_bgopta_and_dtemop/results/phase2_frozen_oneshot_complete_eval.json`
- `antigravity/v3_bgopta_and_dtemop/results/phase2b_frozen_multigranular_and_clean_latency.json`
- `antigravity/v3_bgopta_and_dtemop/figures/fig1_igd_vs_bgopta_mechanism_and_pareto.png`
- `antigravity/v3_bgopta_and_dtemop/figures/fig2_mask_purity_repaired_zenodo_and_vault.png`
- `reports/2026-11-26/antigravity-bgopta-and-final-audit.md`

## 验证命令与结果
- 纯源域零泄漏训练锁定：`python phase1_source_only_train_and_select.py` -> 退出码 `0`，生成 3 种子冻结权重并通过 SHA-256 锁定。
- 完全冻结五大评测模块：`python phase2_frozen_oneshot_eval.py` -> 退出码 `0`，SHA-256 核验通过，输出全部 3 种子明细。
- 七档零冗余测速与多粒度对比：`python phase2b_frozen_multigranular_and_clean_latency.py` -> 退出码 `0`，历史权重 SHA-256 核验通过。
- 仓库合规校验：`python scripts/validate_repository.py --changed-files <tmp>` -> `仓库校验通过。`

## 未完成或阻塞
- 无代码或实验阻塞。自采常山胡柚 190 张图像因尚未经植保专家人工逐框复核真值，严格按项目红线不计入定量准确率报告。
- 所有视觉模型对黄化/斑驳症状的识别均严格界定为田间疑似症状初筛，不等同于 qPCR 病原学确诊。

## 对下游的影响
- 为硕士 SCI 小论文提供了**无测试集泄漏争议、无冗余复杂模块拖累、兼具正反双向机理证明（`IGD-DCR` vs `BG-OPTA`）与实时速度优势（`53.7 FPS`）**的完整实验证据链；且全部文件严格限定于 `antigravity/` 与独立报告内，不影响 GPT 或 DeepSeek 分支工作。

## 下一步建议
- 直接围绕 `IGD-DCR`（输入隔离多粒度目标—环境差分 + 锚点 14 类分布恢复与定位置信度解耦校准）组织小论文正文，将单趟 `BG-OPTA` 作为核心讨论/消融章节展示“为何 ViT 全局自注意力前向污染决定了必须采用输入级空间隔离”，并按 GPU 实时（`IGD-DCR-Batch`）与边缘省算力（`IGD-DCR-Esc`）给出清晰部署建议。
