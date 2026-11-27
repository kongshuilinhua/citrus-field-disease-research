# 四周收尾计划第一至四阶段可立即执行任务全量推进报告

## 任务

遵循“方法与底层 12 份权重 `SHA-256` 完全冻结、绝不伪造人工专家标注、不向 GitHub 上传私有实拍照片、不修改 `coordination/` 及其他助手目录”的原则，一次性完成四周计划中除“等待人工肉眼完成 190 张实拍双盲标注本身”之外的全部实验补齐、工具包构建与论文初稿写作任务。

## 实际完成

1. **第一阶段与外部公开矩阵零缺口收尾**：在 `52ee3b5` 已完成 `SciDB` 统计收尾的基础上，补齐外部三大公开数据集（`Zenodo 11148332` `519` 张、`CitrusUAT` `443` 张对齐单叶、`Sweet Orange V1` `1,389` 张去重留出集）上 `G2_blur`、`A4b`、`A1`、`C3_equal`、`C3_frozen`、`C5a` 的 3 随机种子逐种子明细与 `B = 1,000` 配对检验，并对 `CitrusUAT` 室内单叶负结果执行 $w_{\text{det}} = 0$ 定量归因验证（证实 `-3.79` 个百分点负迁移来自田间检测器分类先验室内分布外偏移，令 $w_{\text{det}} = 0$ 后恢复至 `74.26%`）。
2. **第二阶段双人独立盲标与第三人仲裁全套基础设施**：生成 `190` 张（`148` 个采样隔离簇）权威主索引 CSV，修复跨批次同名文件（`IMG_20260722_101152.jpg`、`IMG_20260722_110142.jpg`）簇映射碰撞；在仓库外本地目录 `D:\新版柑橘\胡柚190张_双人独立标注与仲裁工作包_20261127\` 生成字节级剥离模型预测的 `246` 个纯几何候选框双盲工作包（双人 CSV + 空白仲裁表 + `190` 份 X-AnyLabeling JSON）；编写并通过 $\kappa \ge 0.70$ 门禁与仲裁真值导出脚本自检；输出可直接转发合作公司的胡柚、香柚现有数据 6 项核实函。
3. **第三阶段解盲前无标签部署实测与一次性解盲判决门禁**：对 `180` 张纯净自然 `12MP`（`3072×4096`）衢州胡柚手机实拍图完成同会话严格 `FP32` 端到端推理延迟（含磁盘 JPEG 解码对比）、颈部马氏距离域偏移及 `23` 个多拍对象簇（含 `001/002/091` 同叶三连拍）跨视角预测自洽率实测（`G2_blur` 在 `12MP` 原图上耗时 `88.40 ms`，较双尺度节省 `37.9%`；`C3_equal` 多拍自洽率达 `79.41%` vs `G2_blur` `60.78%`）；编写并通过一次性解盲与预注册三分支判决脚本（`Gate A / Gate B / Gate C`）干跑验证。
4. **第四阶段完整六章中文学术论文初稿**：按农业工程类期刊规范完成完整六章中文论文初稿撰写（逐种子 `[S0, S1, S2]` + 均值、正负机制互证、完整参考文献，仅表 7 保留唯一人工仲裁真值回填槽位）。

## 证据与来源

- 权威主索引：`antigravity/v6_stage2to4_orchard_kit_and_paper_draft/configs/orchard190_148clusters_master_index.csv`（源自 `DS执行_真实照片场景核查_20261124/03_逐图核查/逐图场景核查_190张.csv` 及 `D:\病虫害论文\自采数据\胡柚田间_20260722\raw` 实盘哈希与尺寸核验）
- 全量实验记录 JSON：`antigravity/v6_stage2to4_orchard_kit_and_paper_draft/results/stage2to4_external_and_orchard190_preunblind_complete.json`
- 综合配图：`antigravity/v6_stage2to4_orchard_kit_and_paper_draft/figures/fig4_external_with_negatives_and_huyou190_deployment.png`
- 仓库外本地纯几何盲标工作包：`D:\新版柑橘\胡柚190张_双人独立标注与仲裁工作包_20261127\`（已执行零模型类别字符串断言校验）

## 修改文件

- `antigravity/README.md`
- `antigravity/v4_orchard_candidate_arsenal/scripts/orchard_plug_and_play_cli.py`
- `antigravity/v5_c3_main_method_and_ablation/scripts/orchard_plug_and_play_cli.py`
- `antigravity/v6_stage2to4_orchard_kit_and_paper_draft/configs/orchard190_148clusters_master_index.csv`
- `antigravity/v6_stage2to4_orchard_kit_and_paper_draft/docs/01_第二阶段_190张双人独立盲标规范_Kappa质控与仲裁操作手册.md`
- `antigravity/v6_stage2to4_orchard_kit_and_paper_draft/docs/02_向合作公司核实胡柚香柚现有数据6项关键信息函.md`
- `antigravity/v6_stage2to4_orchard_kit_and_paper_draft/docs/03_第四阶段_完整中文论文初稿_跨背景柑橘症状检测后验校准与背景抑制研究.md`
- `antigravity/v6_stage2to4_orchard_kit_and_paper_draft/figures/fig4_external_with_negatives_and_huyou190_deployment.png`
- `antigravity/v6_stage2to4_orchard_kit_and_paper_draft/results/stage2to4_external_and_orchard190_preunblind_complete.json`
- `antigravity/v6_stage2to4_orchard_kit_and_paper_draft/scripts/generate_stage2to4_figures.py`
- `antigravity/v6_stage2to4_orchard_kit_and_paper_draft/scripts/run_stage2_3_4_complete_external_and_orchard190_suite.py`
- `antigravity/v6_stage2to4_orchard_kit_and_paper_draft/scripts/stage2_kappa_arbitration_and_coco_builder.py`
- `antigravity/v6_stage2to4_orchard_kit_and_paper_draft/scripts/stage3_oneshot_unblind_and_decision_gate.py`
- `reports/2026-11-27/antigravity-stage1to4-complete-kit-external-and-paper-draft.md`

## 验证命令与结果

- `python run_stage2_3_4_complete_external_and_orchard190_suite.py`：退出码 `0`（耗时 `516.4s`，12 份权重 `SHA-256` 吻合，`Stage2=True | Stage3=True` 自检通过）。
- `python generate_stage2to4_figures.py`：退出码 `0`，配图无文字重叠。
- `python orchard_plug_and_play_cli.py --mode check_manifest`：退出码 `0`，6 模型预注册清单与哈希校验通过。
- `python scripts/validate_repository.py --changed-files <staged_list>`：`仓库校验通过。`

## 未完成或阻塞

- `190` 张衢州胡柚自采照片的人工专家真值尚未产生（属于必须由两位真实标注人与第三位仲裁人独立肉眼完成的人工环节，严禁由 AI 代填或伪造真值）；论文初稿表 7 已预留唯一回填槽位。

## 对下游的影响

- 两位标注人可直接使用本地 `D:\新版柑橘\胡柚190张_双人独立标注与仲裁工作包_20261127\` 开展盲标，无需再编写任何数据转换代码；
- 合作公司数据对接可直接发送 `02_向合作公司核实胡柚香柚现有数据6项关键信息函.md`；
- 人工仲裁完成后仅需运行 1 条 `unblind_once` 命令即可自动生成论文表 7 与对应判决结论段。

## 下一步建议

1. 将 `02_向合作公司核实胡柚香柚现有数据6项关键信息函.md` 发给合作公司核实现有胡柚、香柚数据规模与授权边界；
2. 按 `01_第二阶段_190张双人独立盲标规范_Kappa质控与仲裁操作手册.md` 完成 `246` 个纯几何候选框双人独立盲标（首日先做 20 张对齐边界，全量通过 $\kappa \ge 0.70$ 门禁后由第三人仲裁分歧框）；
3. 仲裁真值导出后运行唯一一次 `stage3_oneshot_unblind_and_decision_gate.py --mode unblind_once`，将输出表格填入中文论文初稿表 7 即可进入导师审阅与投稿润色。
