# Antigravity G1–G4 同几何灰环与模糊外环机制实验、A4b 同会话测速、口径修正与果园评测加固报告

## 任务
根据最新代码与方法学审查收敛意见，保持 `C3`（`[0.30, 0.20, 0.50]`）为唯一冻结主方法、零新模型训练、12 份底层权重 `SHA-256` 零改动，一次性完成以下 5 项任务：
1. **修正 `TCD-MGE` 延迟比较口径，补齐 `A4b (1.25× + HFlip)` 同硬件同协议端到端测速**；
2. **按受控 `G1–G4` 设计并附加高斯模糊外环对照（`G2_blur / G4_blur`）完成同几何背景干预机制实验**，通过 $\Delta_{\mathrm{real}} = \text{AP}(G3)-\text{AP}(G1)$、$\Delta_{\mathrm{gray}} = \text{AP}(G4)-\text{AP}(G2)$、$\Delta_{\mathrm{blur}} = \text{AP}(G4_{\mathrm{blur}})-\text{AP}(G2_{\mathrm{blur}})$ 及 `B = 1,000` 逐图配对 Bootstrap 双重差分（DiD）检验背景干扰与尺度互补机制；
3. **补齐 `A1–A5` 全部配对 Bootstrap 95% 置信区间，统一修正论文尺度机制与单自由度权重表述**（主表保留冻结 `C3`，等权 `C3_equal` 放入消融表，严谨界定 `A4b` 与第三尺度 `A5` 的适用边界）；
4. **在 190 张真实果园标签揭晓前将 `A4b` 预先纳入冻结外部验证补充对照（形成 5 模型预声明清单）**，并验证 `A4b` 在线/离线一致性；
5. **修复真实果园评测 CLI 的一对一（1-to-1）置信度优先匹配、专家“不确定”标注自动隔离与按采集簇（Cluster-level）同步配对 Bootstrap**。

## 实际完成
1. **完成受控同几何 `G1–G4` 灰环 + `G2_blur / G4_blur` 高斯模糊外环机制实验与双重差分（DiD）检验**：
   - 验证 `1.25×` 裁剪几何尺寸与 `1.00×` 检测框内部像素在 `real / gray128 / blur31` 下 100% 逐字节恒等；统计确认因叶片长宽比非正方形，`1.25×` 的 `224×224` 视图中框外外环区域平均占总面积的 **`43.76%`（`testB`）/ `49.70%`（`testC`）**。
   - **复杂自然冠层 `SciDB testB`**：`G1`（`1.25×` 原背景 `78.08±1.91%`）→ `G2`（灰环 `78.75±1.10%`, `+0.626%`, $p=0.156$）→ **`G2_blur`（高斯模糊外环 `79.09±1.59%`, `+1.006%`, 95% CI `[+0.129%, +1.870%]`, $p=0.024$ 显著提升；纯 `DINOv2` 从 `76.22%` 升至 `77.91%`）**；同时双尺度增益由原始背景 $\Delta_{\mathrm{real}} = +1.128\%$（$p<0.001$）同步大幅收缩至灰环 $\Delta_{\mathrm{gray}} = +0.148\%$（$p=0.558$）与模糊外环 $\Delta_{\mathrm{blur}} = +0.238\%$（$p=0.294$），**双重差分显著为负：$\text{DiD}_{\mathrm{gray}} = -0.980\%$（95% CI `[-1.660%, -0.288%]`, $p<0.001$）、$\text{DiD}_{\mathrm{blur}} = -0.889\%$（95% CI `[-1.607%, -0.179%]`, $p=0.014$）**，缩减幅度达 `79%–87%`。
   - **跨场景依赖性（`testC` 与 `val`）**：在无邻叶杂乱遮挡的均匀白板 `testC`（`G2 - G1 = +0.261%, p=0.558`；`G2_blur - G1 = +0.118%, p=0.728`）与干净源域 `val`（`-0.232% / -0.171%`）上，框外干预无单尺度跃升，客观印证 `C3` 的背景剥离收益专门针对复杂田间多叶冠层场景。
2. **完成同会话 9 模式端到端公平测速（补齐 `A4b` 与 `C1`）并彻底纠正 `TCD-MGE` 跨协议延迟口径**：
   - 同一 `RTX 4060 Ti` 严格 `FP32` 会话实测：`A4b (1.25×+HFlip)` 为 `val` 均值 **`18.67 ms`**（`P95 = 28.40 ms`，多目标 `≥3框` `28.35 ms`）、`testC` 均值 **`16.78 ms`**（`P95 = 18.98 ms`，实时核验 `Seed 0 AP50–95 = 86.36%` 零误差）；主方法 **`C3 (0.82×+1.25×)`** 为 `val` **`18.98 ms`**（`P95 = 28.97 ms`，多目标 `28.74 ms`）、`testC` **`17.18 ms`**（`P95 = 19.70 ms`，实时核验 `86.53%` 零误差）。
   - 删除跨运行混用的 `TCD-MGE` 延迟比较，统一注明 `v3` 独立协议记录（`val` 均值 `27.53 ms` / `P95 45.75 ms`；`testC` 均值 `25.79 ms` / `P95 29.90 ms`）仅作历史背景参考，不作严格跨协议速度优势证据。
3. **补齐配对 Bootstrap 置信区间并统一论文主方法与机制表述**：
   - 保留冻结 `C3 [0.30, 0.20, 0.50]` 为主表唯一主方法，将零调参等权 `C3_equal [0.30, 0.35, 0.35]`（`78.83% / 84.58%`）放入消融表，明确固定 $w_{\text{det}}=0.30$ 后视觉权重仅有 1 个独立自由度，`C3 − C3_equal` 在 `testB / testC` 为 `+0.358% (p=0.100)` / `+0.374% (p=0.206)` 无显著调权依赖，且 `C3_equal` 在 `testB` 仍显著优于单尺度 `A1`（`+0.776%, p=0.042`）与 `A2`（`+1.008%, p=0.002`）。
   - 严谨界定 `A4b`（`testB` `C3` 显著胜出 `+1.367%, p<0.001`，`testC` `+0.295%, p=0.414` 无显著差异）及 `A5 (+2.50×)`（在当前零额外学习头的冻结线性分类器与后验融合框架下未带来稳定正收益，而非泛化否定第三尺度）。
4. **完成真实果园预声明 5 模型协议修订及一对一匹配 + 按采集簇配对 Bootstrap 修复验证**：
   - 在 `c3_frozen_manifest.json` 揭盲前修订记录中增补 `A4b`（共 5 模型：`YOLOv8n / Full DINOv2 / C1 / A4b / C3`），验证 `A4b` 在线 CLI 与离线参考在 `Seed 0 testB / testC` 差值为 **`0.00%`**；
   - 修复 `orchard_plug_and_play_cli.py` 为严格一对一置信度降序匹配、完备五分法 GT 框归因、专家 `uncertain` 疑难标注隔离及按采集簇同步配对 Bootstrap (`B = 1,000`)，并通过重叠双叶单测与 6 连拍簇冒烟验证。

## 证据与来源
- 修订版冻结配置清单：`antigravity/v5_c3_main_method_and_ablation/configs/c3_frozen_manifest.json`（同步至 `v4`）
- Stage 2b 机制实验、补充 CI、同会话 `A4b` 测速与单测脚本及 JSON：`antigravity/v5_c3_main_method_and_ablation/scripts/run_stage2b_gray_ring_mechanism_and_a4b_speed.py`、`results/stage2b_gray_ring_mechanism_and_a4b_speed.json`
- 更新版 Stage 2 完整 JSON：`antigravity/v5_c3_main_method_and_ablation/results/stage2_c3_controlled_scale_ablation_complete.json`
- 加固定型版果园评测 CLI：`antigravity/v5_c3_main_method_and_ablation/scripts/orchard_plug_and_play_cli.py`（同步至 `v4`）
- 总报告与专题报告：`antigravity/v5_c3_main_method_and_ablation/docs/01_C3主方法锁定_一致性核验与受控双尺度消融报告.md`、`docs/02_G1至G4灰环与模糊外环背景干预机制实验及口径修正专题报告.md`
- 三张无重叠论文级配图：`figures/fig1_c3_controlled_scale_ablation_and_bootstrap.png`、`figures/fig2_c3_fair_end2end_latency_and_pareto.png`、`figures/fig3_g1_to_g4_gray_and_blur_ring_mechanism.png`

## 修改文件
- `antigravity/README.md`
- `antigravity/v4_orchard_candidate_arsenal/configs/c3_frozen_manifest.json`
- `antigravity/v4_orchard_candidate_arsenal/docs/01_面向真实果园落地的两层八候选预注册方法库与保形风险控制指南.md`
- `antigravity/v4_orchard_candidate_arsenal/scripts/orchard_plug_and_play_cli.py`
- `antigravity/v5_c3_main_method_and_ablation/configs/c3_frozen_manifest.json`
- `antigravity/v5_c3_main_method_and_ablation/docs/01_C3主方法锁定_一致性核验与受控双尺度消融报告.md`
- `antigravity/v5_c3_main_method_and_ablation/docs/02_G1至G4灰环与模糊外环背景干预机制实验及口径修正专题报告.md`
- `antigravity/v5_c3_main_method_and_ablation/figures/fig1_c3_controlled_scale_ablation_and_bootstrap.png`
- `antigravity/v5_c3_main_method_and_ablation/figures/fig2_c3_fair_end2end_latency_and_pareto.png`
- `antigravity/v5_c3_main_method_and_ablation/figures/fig3_g1_to_g4_gray_and_blur_ring_mechanism.png`
- `antigravity/v5_c3_main_method_and_ablation/results/stage2_c3_controlled_scale_ablation_complete.json`
- `antigravity/v5_c3_main_method_and_ablation/results/stage2b_gray_ring_mechanism_and_a4b_speed.json`
- `antigravity/v5_c3_main_method_and_ablation/scripts/generate_c3_ablation_figures.py`
- `antigravity/v5_c3_main_method_and_ablation/scripts/orchard_plug_and_play_cli.py`
- `antigravity/v5_c3_main_method_and_ablation/scripts/run_stage2b_gray_ring_mechanism_and_a4b_speed.py`
- `reports/2026-11-27/antigravity-gray-ring-mechanism-a4b-speed-and-orchard-eval-fix.md`

## 验证命令与结果
- 冻结配置与 12 份权重 `SHA-256` 校验：`python antigravity/v5_c3_main_method_and_ablation/scripts/orchard_plug_and_play_cli.py --mode check_manifest` -> `[冻结配置校验通过]`（退出码 `0`）
- Stage 2b 全套受控实验、补充 CI、同会话 `A4b` 测速与果园评测器单测：`python antigravity/v5_c3_main_method_and_ablation/scripts/run_stage2b_gray_ring_mechanism_and_a4b_speed.py` -> `A4b` 在线/离线 `0.00%` 零差，1-to-1 与 6 簇 Bootstrap 单测通过，全部实验完成（退出码 `0`）
- 三张无重叠论文配图重绘与视觉核验：`python antigravity/v5_c3_main_method_and_ablation/scripts/generate_c3_ablation_figures.py` -> 退出码 `0`
- 仓库合规校验：`python scripts/validate_repository.py --changed-files <tmp>` -> `仓库校验通过。`

## 未完成或阻塞
- 公开数据集上的方法效果、受控背景干预机制检验、同会话公平测速与果园评测工具加固已 100% 完成；唯一外部依赖为 190 张真实胡柚/香柚果园照片的植保专家独立框级标注、不确定样本标记与连拍采集簇编号（可立即并行推进）。

## 对下游的影响
- 论文“方法效果 + 受控背景干预机制分析 + 真实果园外部验证”三支柱中的前两项公开实验与统计检验已全部闭环，且真实果园 5 模型一对一按簇评测管线已完备就绪，专家标注完成后即可一键运行 `--mode eval_frozen5` 产出最终盲测表与错误归因。

## 下一步建议
- 保持 `C3` 唯一主方法与 `c3_frozen_manifest.json` 冻结不变，并行启动 190 张果园照片的专家独立标注（对无法仅凭外观确诊样本标记 `uncertain` 并记录同树连拍 `cluster_id`），标注完成后运行 `orchard_plug_and_play_cli.py --mode eval_frozen5` 完成外部验证章节。
