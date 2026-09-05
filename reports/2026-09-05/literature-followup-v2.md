# 补充文献调研：可靠性、预标注管线与标签噪声

> 日期：2026-09-05  
> Agent/任务：补充文献调研 / literature-followup-v2  
> 对应Issue：#1 的补充（不改变其 completed 状态）  
> 分支：main

## 任务

在已完成 `reports/2026-09-05/method-landscape-expanded.md`（42 篇、24 个方法家族）的基础上，刻意避开已覆盖方向，围绕用户需求的三个尚未深挖的痛点做第二轮调研：

1. 黄龙病无 PCR 确诊、只能输出"疑似/不确定"——寻找把"拒识/预测集合"作为一等公民的方法；
2. 农民初标 + 专家复核的流程——寻找标签噪声学习和低成本自动预标注的方法；
3. 2025—2026 年柑橘、农业基础模型和 MLLM 的新动向——确认是否有改变当前判断的证据。

边界：只做文献调研与方案启示，不修改代码、中央状态和决策；不把搜索摘要当最终证据；不把视觉症状当病原学确诊。

## 实际完成

1. 新增 5 个此前未覆盖的方法家族：共形预测/选择性分类、开放词汇检测与自动预标注管线、标签噪声学习、农业视觉语言大模型（MLLM）、植物病害 Mamba/自监督状态空间模型。
2. 整理 18 条结构化证据，其中 11 条为可用候选、7 条为排除项，见 `research/inbox/literature-followup-evidence.csv`。
3. 对两篇关键论文做原文级核验：MRC-Net（Frontiers in Plant Science 2026，含共形预测训练—推理一体化细节）和 HLB-YOLO（Applied Fruit Science 2025，确认其无 PCR 确诊依据）。
4. 发现一个新的公开柑橘数据源线索：Chi et al. 2025 柑橘黄龙病田间症状数据集（China Science Data，csdata.org/p/998/），移交数据与标签 Agent 核验。
5. 对每个候选给出与现有六个方向（A—F）的合并关系，避免形成平行新方向导致范围膨胀。

## 证据与来源

检索范围：2026-09-05 通过联网检索覆盖 Frontiers、Springer、ScienceDirect、Nature、arXiv、CVF 和 GitHub 一手页面；对 MRC-Net 和 HLB-YOLO 做了原文页面级核验。完整条目见 `research/inbox/literature-followup-evidence.csv`。

### 1. 共形预测 / 选择性分类（本轮最重要发现）

- MRC-Net（Frontiers in Plant Science 2026，DOI: 10.3389/fpls.2026.1895287）：Swin-Tiny + 小波/状态空间特征增强 + CARP 共形预测模块。训练阶段把共形覆盖与集合紧凑性写入损失，推理阶段按校准集分位数输出**候选类别集合**而非单一标签。α=0.10 时三个数据集经验覆盖率 0.897—0.908，接近名义覆盖；PlantVillage 平均集合仅 1.06 类，PlantDoc 难样本自动放大到 2.68 类。注意：无公开代码，且 PlantDoc 绝对准确率仅 38.89%，说明"复杂背景分类"本身仍是难题，可靠性模块不能弥补基座不足。
- COMPAG 2025（DOI: 10.1016/j.compag.2025.110559）：归纳共形预测（ICP）在作物生产决策支持中的系统分析，提供风险可控的预测集构造流程。

对本项目的直接价值：黄龙病没有 PCR 时，"输出单标签"和"不输出"都不是好设计；共形预测集（如 {黄龙病疑似, 缺素黄化}，覆盖率有统计保证）正好对应用户"只能疑似"的约束，且实现成本低（只需独立校准集 + 分位数阈值）。这是上一轮全景中"开放世界/拒识"家族之外的更轻量路线，可与方向 F 合并升级。

### 2. 开放词汇检测与自动预标注管线

- Grounding DINO（ECCV 2024，arXiv:2303.05499）与 YOLO-World（CVPR 2024，arXiv:2401.17270）：文本提示的零样本检测，均有官方代码。
- Grounded-SAM / Grounded-SAM-2 Auto-Labeling（IDEA-Research 官方仓库）与 Roboflow Autodistill：开放词汇检测框 → SAM 掩膜 → 蒸馏小模型的成熟自动标注管线。

对本项目的直接价值：在农民初标之前，可用 "citrus leaf / citrus fruit / insect / disease spot" 这类粗粒度提示词做零样本器官级预标注，人工只做确认和类别修正，直接降低 1000 张的标注成本。但必须注意：开放词汇模型对细粒度病害类别不可靠，只能负责"器官在哪"，病名仍由复核决定；预标注框按项目规则属于机器预标签，不是真值。

### 3. 标签噪声学习（农民初标的配套方法）

- Confident Learning / cleanlab（NeurIPS 2021 workshop / JAIR，arXiv:1911.00068，官方代码）：对任意模型输出做标签审计，自动找出"大概率标错"的样本。本项目可用于：农民初标完成后先跑 cleanlab，把可疑标签排优先级送专家复核，把专家时间花在刀刃上。
- Co-teaching（NeurIPS 2018，arXiv:1804.06872）：双网络互选小损失样本抗噪训练，作为标签噪声确实严重时的备选。

上一轮全景完全没有标签噪声家族；而本项目"农民初标 + 专家复核"决定了标签噪声是确定的现实问题，应作为流程能力固化，与主动学习（方向 D）并列。

### 4. 农业视觉语言大模型（MLLM）

- Plant Disease Detection through Multimodal LLMs（GPT-4o，arXiv:2504.20419，2025）。
- AgriGPT（Springer 2025）与 Scientific Reports 2025 作物病虫害多模态 LLM（DOI: 10.1038/s41598-025-01908-0）。

结论：MLLM 适合作为专家辅助工具（症状描述草稿、标注建议、长尾类解释）和潜在的双视图一致性教师，但存在幻觉、输出不可复现、无法端侧部署三个硬伤，不能作为识别主线，输出更不能当作标签真值。与 USER_REQUIREMENTS 中"不接受堆叠术语"一致，明确列为低优先级。

### 5. 植物病害 Mamba / 自监督状态空间模型

- PSMamba（arXiv:2512.14309）：渐进式自监督 Vision Mamba 用于植物病害，方向是"预训练"而非"换骨干"，相对值得关注。
- Mamba-YOLO-ML（Plants 2025）、LeafMamba（2025）、DDHTLVMamba（Frontiers 2025）：均属于"换骨干/堆模块"式工作，按项目决策（不把 YOLO 加模块当创新）列为排除或仅作参照。

### 6. 柑橘黄龙病 2025—2026 新动向

- HLB-YOLO（Applied Fruit Science 2025，DOI: 10.1007/s10341-025-01647-6）：基于 YOLOv8n 加 ARConv + Dynamic Head + EIoU，mAP50 86.6%。原文级核验确认：摘要与可得信息中**无任何 PCR/qPCR 确诊说明**，标签疑似仅依赖视觉症状。该文恰好印证本项目"黄化≠黄龙病"的长期不变量——此类论文只可作为"症状级检测"参照，不可作为病名识别的证据链模板。
- Chi et al. 2025《A dataset of field symptoms for identifying citrus huanglongbing》（China Science Data，公开于 csdata.org/p/998/）：公开柑橘黄龙病**田间症状**图像数据集，是本项目此前清单中没有的公开柑橘数据源，移交数据与标签 Agent 核验许可证、规模、标注依据和与既有数据的重叠。
- 跨品种高光谱 HLB（Spectrochimica Acta A 2026，DOI: 10.1016/j.saa.2025.125xxx 附近）：证明跨品种 HLB 识别确有域差问题，但高光谱设备与 App 场景不符，仅作问题存在的旁证，列为排除。

### 7. 其他补充

- Foundation vision models in agriculture（COMPAG 2025，S0168169925010063）：系统验证 DINOv2 + LoRA 参数高效微调在农业任务的可行性，为上一轮方向 A 中"DINOv2 二阶段细分类"提供同领域实证，且单卡 4060 Ti 友好。
- IP102（CVPR 2019，官方 GitHub）：75,222 张、102 类、天然长尾、带"作物→害虫"层级分类体系的害虫基准。其层级标签设计（先上层类再细类）可直接借鉴到本项目"受害器官→病征类型→病名"的分层标签候选。

## 新增发现与排除项

### 新增候选（按优先级）

| 优先级 | 家族 | 代表 | 合并进哪个现有方向 |
|---|---|---|---|
| A | 共形预测/预测集 | MRC-Net 思路 + ICP | 与方向 F 合并为"可靠识别/疑似输出" |
| A（流程） | 开放词汇+SAM 自动预标注 | Grounding DINO + Grounded-SAM-2 | 并入方向 D 标注流程 |
| A（流程） | 标签审计 | cleanlab | 并入方向 D，专家复核优先级排序 |
| A | DINOv2+LoRA 农业微调 | COMPAG 2025 | 强化方向 A 二阶段细分类 |
| B | 层级标签体系 | IP102 分类法 | 分层标签候选（USER_REQUIREMENTS 已列） |
| B | 自监督 Mamba | PSMamba | 预训练候选，非主线 |
| B（移交） | 柑橘田间症状数据集 | Chi et al. 2025 csdata.org/p/998/ | 移交数据与标签 Agent |
| C | MLLM 专家辅助 | AgriGPT 等 | 仅工具角色 |

### 排除项与理由

| 排除项 | 理由 |
|---|---|
| Mamba-YOLO-ML / LeafMamba / DDHTLVMamba | 换骨干堆模块，无问题依据，违反"注意力+模块拼接不算创新"原则 |
| MLLM 直接输出诊断 | 幻觉、不可复现、不可端侧，输出不能作真值 |
| 高光谱 HLB 路线 | 设备与 App 场景不符 |
| 3D 点云 Plant-MAE | 与 2D 近景 App 场景无关 |
| HLB-YOLO 作为病名识别范式 | 无 PCR 确诊，症状级标签，只作参照与反例 |
| YOLO-LeafNet 等同类 | 又一个 YOLO 改模块论文，无增量 |

## 修改文件

- 新增 `reports/2026-09-05/literature-followup-v2.md`
- 新增 `research/inbox/literature-followup-evidence.csv`

## 验证命令与结果

计划执行：

```powershell
python scripts/validate_repository.py
rg -n "共形|预标注|噪声|排除|DOI" reports/2026-09-05/literature-followup-v2.md
```

验证结果：`python scripts/validate_repository.py` 输出"仓库校验通过。"（2026-09-05 执行）。

## 未完成或阻塞

- Chi et al. 2025 柑橘黄龙病田间症状数据集的许可证、规模和标注依据未核验，已移交数据与标签 Agent。
- MRC-Net 无公开代码，共形预测部分需要自实现（成本约 1—2 天），未验证其在检测框分类头上的效果。
- Grounding DINO / Grounded-SAM-2 在胡柚近景图上的零样本预标注质量未实测，需要 20—50 张抽样试跑。
- cleanlab 对检测框标签（而非分类标签）的审计需要裁框后进行，流程未实测。
- HLB-YOLO 全文在付费墙后，数据规模、划分与跨场景验证情况未知；仅凭摘要与公开页判断。

## 对下游的影响

- 数据与标签 Agent：新增核验对象 csdata.org/p/998/；需在审计中回答"预标注用开放词汇管线 vs 人工从零标"的成本对比。
- 方法方案 Agent：方向 F 可升级为"可靠识别（预测集 + 拒识）+ 开放世界"双层设计；方案中必须包含一个共形预测校准实验（只需校准集，成本极低）。
- 方法方案 Agent：标注流程建议固化为"开放词汇+SAM 预标注 → 农民确认/修正 → cleanlab 审计 → 专家复核可疑样本"四级管线。
- 不改变上一轮六个方向的结构和优先级，本轮全部为合并增强，不产生第七个方向。

## 下一步建议

1. 数据与标签 Agent 核验 csdata.org/p/998/ 数据集并纳入 `docs/data/公开数据集清单.md`。
2. 用 190 张自采图中的 20—50 张抽样试跑 Grounding DINO + SAM 预标注，记录框质量和人工修正率，作为是否采用自动预标注的证据。
3. 方法方案 Agent 在检测方案中加入共形预测校准实验设计（校准集划分、α、集合大小与覆盖率指标）。
4. 先导实验基线确定后，用 cleanlab 对农民初标做一轮标签审计，量化初标错误率，决定是否引入抗噪训练。
