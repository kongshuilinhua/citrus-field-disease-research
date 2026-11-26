# Antigravity 主方法 C3 冻结一致性核验、受控双尺度消融（A1–A5）与公平测速报告

## 任务
严格执行论文收敛阶段的第一阶段（第 1–2 天）与第二阶段（第 3–5 天）全部工程修复与受控实验任务（零新模型训练）：
1. **第一阶段（修复实验一致性与冻结主方法）**：
   - 建立冻结配置清单 `c3_frozen_manifest.json`，固化全部 12 份底层权重 `SHA-256`、主方法 `C3` 融合权重 `[0.30, 0.20, 0.50]` 及按种子标定的 `C2` 参数 `γ = [0.35, 0.15, 0.35]`，消除 CLI 硬编码与未实现模式声明；
   - 统一 `检测→裁剪→分类→融合→NMS` 在线与离线推理流程，验证 `C3` 在线端到端推理与离线结果逐框一致；
   - 冻结未来真实果园唯一评测四模型（`YOLOv8n`、`Full DINOv2`、`C1`、**`C3`**）及配套指标输出。
2. **第二阶段（完成 `A1–A5` 关键受控尺度消融与公平测速）**：
   - 在相同冻结骨干、零额外 MLP 参数、相同后处理、且消除总检测器权重混杂（固定 `w_det = 0.30`、仅用源域 `SciDB val` 选定视角比例，绝不用 `testB/testC` 选方法）的前提下，完成 `A1 (1.25×)`、`A2 (0.82×)`、`A3/C3 (0.82×+1.25×)`、`A4a (1.15×+1.35× 相近尺度双裁剪 TTA)`、`A4b (1.25×+水平翻转双视角 TTA)`、`A5 (0.82×+1.25×+2.50×)` 的三随机种子对比、`B = 1,000` 逐图配对 Bootstrap 95% 置信区间检验及含多目标图像的端到端公平测速。

## 实际完成
1. **完成第一阶段全部 5 项一致性修复并通过三随机种子全集逐框验收**：
   - 生成并双目录同步 `configs/c3_frozen_manifest.json`，12 份底层权重文件 `SHA-256` 100% 校验通过，CLI 全面改为从冻结清单读取 `C3` 权重 `[0.30, 0.20, 0.50]` 与按种子 `C2` 的 `γ`，并清理未实现模式声明、补齐第四阶段冻结四模型评测模式 `--mode eval_frozen4`。
   - 定位并修复单图在线推理中 `Ultralytics` 默认对非方图启用 `auto=True` 矩形缩放及 `non_max_suppression` 原地将检测头张量前 4 列由 `xywh` 改写为 `xyxy` 的两处底层差异。修复后，3 个随机种子在 `val / testB / testC` 抽检共 **800 个前景框的预测类别不一致数 = `0`**（后验概率最大绝对误差仅 `1e-6 ~ 5e-6`），9 个划分全集端到端实时 `AP50–95` 中 8 个与 `v4` 离线结果实现 `0.00%` 零差（仅 `Seed 1 testB` 为 `78.12%` vs `78.11%`）。
2. **完成第二阶段 `A1–A5` 严格受控尺度消融与 `B = 1,000` 逐图配对 Bootstrap 检验**：
   - **单尺度对比（`A3` vs `A1` / `A2`）**：主方法 `C3 (0.82×+1.25×)` 取得 `testB 79.19±1.10%`、`testC 84.96±1.38%`，在配对 Bootstrap 上显著优于标准单尺度 `A1 (1.25×: 78.08% / 84.29%)`（`testB` `+1.135%`, 95% CI `[+0.549%, +1.769%]`, $p<0.001$；`testC` `+0.650%`, 95% CI `[+0.023%, +1.261%]`, $p=0.044$；对比历史 `C1` 为 `+0.920%`, $p=0.002$ / `+1.118%`, $p<0.001$），亦显著优于紧致单尺度 `A2 (0.82×: 77.78% / 82.47%)`（`+1.366%`, $p<0.001$ / `+2.474%`, $p<0.001$）。
   - **排除普通双视角 TTA 解释（`A3` vs `A4a` / `A4b`）**：在复杂自然田间背景 `SciDB testB` 上，等算力的相近双尺度裁剪 `A4a (1.15×+1.35×)`（`77.70±1.89%`）与水平翻转 `A4b (1.25×+HFlip)`（`77.80±1.56%`）均未超过单尺度 `A1`（`78.08%`）；`C3` 在 `testB` 上显著优于 `A4a` **`+1.488%`（95% CI `[+0.755%, +2.291%]`, $p<0.001$；逐种子 `val` 选权对比亦为 `+1.225%`, $p=0.002$，等权对比 `+0.984%`, $p=0.008$）**，显著优于 `A4b` **`+1.367%`（95% CI `[+0.682%, +2.037%]`, $p<0.001$）**。在无杂叶遮挡的纯白板 `testC` 上，`C3` 仍显著优于 `A4a`（`+0.961%`, $p=0.010$），与 `A4b` 差距缩小且置信区间跨零（`+0.295%`, 95% CI `[-0.414%, +0.983%]`, $p=0.414$），客观印证了 `0.82×` 视角的增益专一性地源于隔离复杂田间背景杂叶。
   - **验证扩大环境视角无稳定正收益（`A5` vs `A3`）**：加入 `2.50×` 的三尺度 `A5`（`78.94±1.34% / 84.48±1.39%`）相对 `C3` 为 **`-0.258%`（$p=0.284$）/ `-0.418%`（$p=0.072$）**，无任何精度增益。
3. **完成包含多目标图像（`≥3框`）与全集实时 AP 验算的端到端公平测速（RTX 4060 Ti, FP32）**：
   - `C3` 在 `SciDB val` 平均延迟为 **`19.17 ms`（`P95 = 30.77 ms`，多目标 `≥3框` 均值 `30.04 ms`）**，在 `SciDB testC` 全集 602 图平均延迟为 **`17.46 ms`（`P95 = 20.96 ms`，测速环路实时验算 `Seed 0 AP50–95 = 86.53%` 与离线零误差）**，较单框校准 `A1`（`16.65 ms`）仅增加 `0.81 ms`，比三尺度 `A5` 在多目标图像上（`41.27 ms`）快 `27.2%`。

## 证据与来源
- 冻结配置清单：`antigravity/v5_c3_main_method_and_ablation/configs/c3_frozen_manifest.json`
- Stage 1 一致性核验脚本与 JSON：`antigravity/v5_c3_main_method_and_ablation/scripts/build_and_verify_c3_manifest_and_parity.py`、`results/stage1_c3_online_offline_parity_verification.json`
- Stage 2 受控尺度消融与测速脚本及 JSON：`antigravity/v5_c3_main_method_and_ablation/scripts/run_stage2_controlled_scale_ablation_and_speed.py`、`results/stage2_c3_controlled_scale_ablation_complete.json`
- 修复定型版统一推理 CLI：`antigravity/v5_c3_main_method_and_ablation/scripts/orchard_plug_and_play_cli.py`（已同步更新 `v4` 同名脚本）
- 论文方法公式与消融总报告：`antigravity/v5_c3_main_method_and_ablation/docs/01_C3主方法锁定_一致性核验与受控双尺度消融报告.md`
- 论文配图：`antigravity/v5_c3_main_method_and_ablation/figures/fig1_c3_controlled_scale_ablation_and_bootstrap.png`、`figures/fig2_c3_fair_end2end_latency_and_pareto.png`

## 修改文件
- `antigravity/README.md`
- `antigravity/v4_orchard_candidate_arsenal/configs/c3_frozen_manifest.json`
- `antigravity/v4_orchard_candidate_arsenal/docs/01_面向真实果园落地的两层八候选预注册方法库与保形风险控制指南.md`
- `antigravity/v4_orchard_candidate_arsenal/scripts/orchard_plug_and_play_cli.py`
- `antigravity/v5_c3_main_method_and_ablation/configs/c3_frozen_manifest.json`
- `antigravity/v5_c3_main_method_and_ablation/docs/01_C3主方法锁定_一致性核验与受控双尺度消融报告.md`
- `antigravity/v5_c3_main_method_and_ablation/figures/fig1_c3_controlled_scale_ablation_and_bootstrap.png`
- `antigravity/v5_c3_main_method_and_ablation/figures/fig2_c3_fair_end2end_latency_and_pareto.png`
- `antigravity/v5_c3_main_method_and_ablation/results/stage1_c3_online_offline_parity_verification.json`
- `antigravity/v5_c3_main_method_and_ablation/results/stage2_c3_controlled_scale_ablation_complete.json`
- `antigravity/v5_c3_main_method_and_ablation/scripts/build_and_verify_c3_manifest_and_parity.py`
- `antigravity/v5_c3_main_method_and_ablation/scripts/generate_c3_ablation_figures.py`
- `antigravity/v5_c3_main_method_and_ablation/scripts/orchard_plug_and_play_cli.py`
- `antigravity/v5_c3_main_method_and_ablation/scripts/run_stage2_controlled_scale_ablation_and_speed.py`
- `reports/2026-11-26/antigravity-c3-main-method-lock-and-scale-ablation.md`

## 验证命令与结果
- Stage 1 在线/离线逐框与全集一致性核验：`python antigravity/v5_c3_main_method_and_ablation/scripts/build_and_verify_c3_manifest_and_parity.py` -> 12 份权重哈希通过，800 抽检框 `0` 类别误差，9 划分全集实时推理核验通过（退出码 `0`）
- CLI 冻结清单校验：`python antigravity/v5_c3_main_method_and_ablation/scripts/orchard_plug_and_play_cli.py --mode check_manifest` -> `[冻结配置校验通过]`（退出码 `0`）
- Stage 2 受控尺度消融与公平测速：`python antigravity/v5_c3_main_method_and_ablation/scripts/run_stage2_controlled_scale_ablation_and_speed.py` -> 完成 3 种子消融、Bootstrap 与测速（退出码 `0`）
- 配图生成与视觉核验：`python antigravity/v5_c3_main_method_and_ablation/scripts/generate_c3_ablation_figures.py` -> 退出码 `0`
- 仓库合规校验：`python scripts/validate_repository.py --changed-files <tmp>` -> `仓库校验通过。`

## 未完成或阻塞
- 第三阶段（190 张自采真实果园照片的农学专家框级标注、不确定样本标记、连拍采集簇编号及胡柚/香柚类别与 `SciDB` 14 类映射核对）需外部植保人员参与完成后，方可启动第四阶段冻结四模型盲测与 DS 错误归因统计。

## 对下游的影响
- Gemini 负责的第一阶段（一致性修复与 `C3` 冻结）和第二阶段（`A1–A5` 受控双尺度消融与公平测速）已全部闭环并通过严格检验；论文引言、相关工作、方法（完整公式已整理就绪）与实验设置四章现在即可直接动笔撰写。

## 下一步建议
- 严格保持 `c3_frozen_manifest.json` 冻结不变，停止一切新模块开发或模型重训；可使用 `orchard_plug_and_play_cli.py --mode preannotate --candidate C3` 生成初始辅助框供植保专家修订（对无法判断样本标为“不确定”并记录连拍簇号），待标注与类别对齐确认后运行 `--mode eval_frozen4` 一次性产出第四阶段真实果园结果。
