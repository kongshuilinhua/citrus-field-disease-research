# Antigravity 第一阶段统计收尾（v1.1）：G3−G2/G2_blur 配对检验、零调参等权主表、逐种子勘误与第 6 盲测模型 G2_blur 入列报告

## 任务
严格执行 4 周收敛计划中的“第一阶段：统计收尾（5 项任务）”，不改动已冻结模型权重（12 份 `SHA-256` 锁定），完成：
1. 补齐决定“两刀是否必要”的 `G3 − G2`、`G3 − G2_blur`、`C3 − A4b`（含冻结版与零调参等权版）的 `B = 1,000` 配对 Bootstrap 检验，追加至 `stage2b_gray_ring_mechanism_and_a4b_speed.json`；
2. 更新报告至 `v1.1`：主表改用零调参等权 `C3_equal [0.30, 0.35, 0.35]` 作为主汇报数字（消除视觉权重调参自由度质疑），冻结权重版 `C3_frozen [0.30, 0.20, 0.50]` 置于表注；
3. 全文将“三种子均值 + 逐图 CI”改为“逐种子 `[Seed 0, Seed 1, Seed 2]` + 均值”，在勘误段明确标出 `DiD_gray` 的 Seed 0 为 `+0.505%`（无收缩，收缩系 `2/3` 种子现象）；
4. 在勘误段正式撤回并修正此前“等权口径同样显著收缩”的事实性错误（实测等权 `DiD` 为 `p = 0.122 / 0.098` 不显著）；
5. 在 `c3_frozen_manifest.json` 与果园评测 CLI 中将 `G2_blur`（单刀 `1.25×` + `31×31` 框外高斯模糊遮罩）增补为第 6 个预注册盲测模型，并完成同会话 10 模式公平测速与在线/离线零误差核验。

## 实际完成
1. **完成 `G3 − G2`、`G3 − G2_blur` 与 `C3 − A4b` 的 `B = 1,000` 逐种子及均值配对检验，明确回答核心判决问题**：
   - **核心一句话回答**：第二次裁剪（`0.82×`）相对“单刀 `1.25×` + 框外模糊遮罩（`G2_blur`）”在复杂田间背景 `testB` 上仅相差 **`+0.122%`（冻结 `C3`: 95% CI `[-0.603%, +0.906%]`, `p = 0.7800`，逐种子 `[-0.203%, +0.740%, -0.171%]`）** 或 **`-0.237%`（等权 `C3_equal`: 95% CI `[-1.002%, +0.531%]`, `p = 0.5540`，逐种子 `[-0.026%, -0.027%, -0.659%]`）**；在 `testC` 上相差 `+0.528%`（`p = 0.1340`）或 `+0.144%`（`p = 0.7300`）；对比单刀灰环 `G2` 同样不显著（`testB +0.502%, p = 0.1920` / `等权 +0.143%, p = 0.7180`）。
   - 依据预设判定门（未达 `≥ +0.40pt 且 p < 0.05`），不改动冻结协议，将报告 `v1.1` 的方法与讨论口径改写为：“单刀框外模糊遮罩 `G2_blur` 为最简机制实现（`1.0×` ViT 算力，在 `testB` 上亦显著优于双刀翻转 `A4b` `+1.214%, p = 0.0080`），双尺度 `C3` 为无需显式框边界遮罩的等效增强实现”。
2. **完成报告 `v1.1` 主表切换与两项正式勘误**：
   - 主对比表采用零调参等权 `C3_equal [0.30, 0.35, 0.35]`（`val [90.96, 89.08, 90.85]` 均值 `90.30%`；`testB [80.49, 77.34, 78.67]` 均值 `78.83% > A1 78.08%, p = 0.042`；`testC [85.76, 84.22, 83.76]` 均值 `84.58%`），冻结版 `C3_frozen`（`79.19% / 84.96%`）置于表注；三张配图同步叠加 3 个随机种子的独立散点。
   - **勘误 1**：全文改为逐种子 + 均值并列展示，明确指出 `testB` 上冻结 `C3` 的 `DiD_gray` 逐种子为 **`[+0.505%, -1.989%, -1.455%]`**（`p = [0.370, <0.001, 0.006]`，Seed 0 无收缩）、`DiD_blur` 逐种子为 `[-0.154%, -1.011%, -1.503%]`，增益收缩系 `2/3` 种子现象。
   - **勘误 2**：正式撤回“等权口径同样显著收缩”的错误表述，明确标出等权口径 `DiD_gray_eq = -0.621%`（`p = 0.1220`）、`DiD_blur_eq = -0.688%`（`p = 0.0980`）均未达统计显著。
3. **完成第 6 预注册盲测模型 `G2_blur` 入列、在线一致性核验与 10 模式同会话公平测速**：
   - 在 `c3_frozen_manifest.json` 与 `orchard_plug_and_play_cli.py` 中增补 `G2_blur`（共 6 模型：`YOLOv8n / Full DINOv2 / C1 / G2_blur / A4b / C3`，同时附 `C3_equal`），实测 `G2_blur` 与 `C3_equal` 在 `Seed 0 testB / testC` 上在线 CLI 与离线参考差值均为 **`0.00%`**，并通过按簇 Bootstrap 冒烟测试。
   - 完成全 10 模式同会话端到端测速（`RTX 4060 Ti FP32`）：`A1` `17.06 / 16.47 ms`，`G2_blur` `19.32 / 17.84 ms`（`1.0×` ViT 算力 + CPU `31×31` 高斯模糊），`A4b` `18.61 / 16.75 ms`，`C3` `18.93 / 17.08 ms`，`A5` `22.96 / 19.77 ms`（全部经实时 AP 零误差核验）。

## 证据与来源
- 预注册 6 模型冻结配置清单：`antigravity/v5_c3_main_method_and_ablation/configs/c3_frozen_manifest.json`
- 第一阶段统计收尾与同会话 10 模式测速脚本：`antigravity/v5_c3_main_method_and_ablation/scripts/run_stage2b_v11_statistical_wrapup.py`、`scripts/run_unified_10modes_same_session_latency.py`
- 追加更新后的统计结果 JSON：`antigravity/v5_c3_main_method_and_ablation/results/stage2b_gray_ring_mechanism_and_a4b_speed.json`、`results/stage2_c3_controlled_scale_ablation_complete.json`
- 报告 `v1.1`（含勘误段与逐种子表格）：`antigravity/v5_c3_main_method_and_ablation/docs/01_C3主方法锁定_一致性核验与受控双尺度消融报告.md`、`docs/02_G1至G4灰环与模糊外环背景干预机制实验及口径修正专题报告.md`
- 叠加逐种子散点的三张无重叠配图：`figures/fig1_c3_controlled_scale_ablation_and_bootstrap.png`、`figures/fig2_c3_fair_end2end_latency_and_pareto.png`、`figures/fig3_g1_to_g4_gray_and_blur_ring_mechanism.png`

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
- `antigravity/v5_c3_main_method_and_ablation/scripts/run_stage2b_v11_statistical_wrapup.py`
- `antigravity/v5_c3_main_method_and_ablation/scripts/run_unified_10modes_same_session_latency.py`
- `reports/2026-11-27/antigravity-gray-ring-mechanism-a4b-speed-and-orchard-eval-fix.md`

## 验证命令与结果
- 6 模型冻结配置哈希核验：`python antigravity/v5_c3_main_method_and_ablation/scripts/orchard_plug_and_play_cli.py --mode check_manifest` -> 12 份权重 `SHA-256` 与 6 模型配置通过（退出码 `0`）
- 统计收尾与 `G2_blur / C3_equal` 在线一致性核验：`python antigravity/v5_c3_main_method_and_ablation/scripts/run_stage2b_v11_statistical_wrapup.py` -> 在线/离线 `0.00%` 零差，全部配对 Bootstrap 完成（退出码 `0`）
- 全 10 模式同会话测速与图表生成：`python antigravity/v5_c3_main_method_and_ablation/scripts/run_unified_10modes_same_session_latency.py` & `generate_c3_ablation_figures.py` -> 退出码 `0`
- 仓库合规校验：`python scripts/validate_repository.py --changed-files <tmp>` -> `仓库校验通过。`

## 未完成或阻塞
- 第一阶段 5 项统计收尾已全部完成闭环；下一步关键路径为第二阶段 190 张果园实拍图（148 簇）的双人独立标注与仲裁及公司数据 6 项元信息确认。

## 对下游的影响
- “校准管线主贡献 + 单刀遮罩最简实现 / 双尺度等效增强 + 正负结果互证”的论文证据骨架已完全锁定，无需再修改任何模型或补充公开集实验；真实果园 6 模型盲测工具已就绪，待标注完成后只跑一次即可直接进入初稿撰写。

## 下一步建议
- 按标注规范推进 190 张 / 148 簇双人独立标注与第三人仲裁（目标 Kappa ≥ 0.7，“无法判断”单独成类，不强标 HLB），标注到位后执行一次 `orchard_plug_and_play_cli.py --mode eval_frozen6` 完成盲测。
