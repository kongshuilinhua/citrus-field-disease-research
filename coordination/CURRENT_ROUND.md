# 2026-09-05：跨作物方法与数据可行性并行调研

## Current Judgment

1. 选题尚未锁定，不能先定YOLO、DETR或无监督再寻找问题。
2. 现有自采图适合近景分类、病斑检测或分割，但病名标签尚未完成专家复核。
3. 本轮只做调研和方案收敛，不启动正式大规模训练。

## Round Goal

1. 建立2024—2026年跨作物方法证据表。
2. 确认现有数据能支持的标签和任务。
3. 为下一轮形成分类、检测、分割三套可比较方案。

## Gate Conditions

1. 文献Agent和数据Agent可以并行。
2. 方法Agent必须等待两个上游报告完成。
3. 实验Agent必须等待方法与标签协议稳定。

## Global Forbidden Scope

```text
完整数据集、自采原图和模型权重
coordination/PROJECT_STATE.md（普通Agent只读）
coordination/DECISIONS.md（普通Agent只读）
其他Agent的报告文件
```

## Agent Division

### 跨作物文献Agent

Dispatch Status: ready

#### Work Content

调研2024—2026年水稻、玉米、小麦、番茄、苹果、葡萄、棉花、花生等作物中的监督分类、Faster R-CNN、DETR/RT-DETR、病斑分割、Transformer、Mamba和分类—分割—严重度多任务方法。重点比较真实背景、标签成本、独立跨场景验证和可复现性。

#### Modification Scope

```text
reports/2026-09-05/cross-crop-methods.md
research/inbox/cross-crop-methods.csv
```

#### Forbidden Scope

```text
coordination/PROJECT_STATE.md
coordination/DECISIONS.md
src/
data/
```

#### Dependency Documents

```text
README.md
AGENTS.md
docs/项目接手文档.md
docs/research/全库速览_69篇.md
coordination/PROJECT_STATE.md
coordination/DECISIONS.md
本任务表
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
rg -n "DOI|数据|代码|背景|标注|局限" reports/2026-09-05/cross-crop-methods.md
```

#### Dispatch Prompt

```text
你是本轮跨作物文献Agent。依次阅读README.md、AGENTS.md、docs/项目接手文档.md、coordination/PROJECT_STATE.md、coordination/DECISIONS.md和coordination/CURRENT_ROUND.md。调研2024—2026年其他作物中的监督分类、Faster R-CNN、DETR/RT-DETR、分割、Transformer、Mamba和多任务严重度方法。只使用一手论文、官方代码和官方数据页面。每篇记录任务、作物、背景、规模、标注、方法、对照、跨场景验证、代码/数据可得性、4060 Ti可行性、可迁移点和排除理由。不要预设排除YOLO，也不要把模型换名当创新。只修改reports/2026-09-05/cross-crop-methods.md和research/inbox/cross-crop-methods.csv，完成后运行校验并把结果写入报告。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`，并附候选方法优先级与排除项。

#### Report Result

Waiting.

### 数据与标签Agent

Dispatch Status: ready

#### Work Content

核验现有公开数据与190张自采图能够支持的分类、框、掩膜、严重度和跨拍摄者评测协议。检查许可证、下载状态、同源重复、标签可靠性和数据泄漏风险。

#### Modification Scope

```text
reports/2026-09-05/data-label-audit.md
research/inbox/dataset-audit.csv
```

#### Forbidden Scope

```text
coordination/PROJECT_STATE.md
coordination/DECISIONS.md
src/
完整数据和原始图片
```

#### Dependency Documents

```text
README.md
AGENTS.md
docs/项目接手文档.md
docs/data/公开数据集清单.md
data/DATA_MANIFEST.csv
data/self_collected_prelabels.csv
coordination/PROJECT_STATE.md
本任务表
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
rg -n "许可证|下载|重复|标签|泄漏|复核" reports/2026-09-05/data-label-audit.md
```

#### Dispatch Prompt

```text
你是本轮数据与标签Agent。依次阅读README.md、AGENTS.md、docs/项目接手文档.md、docs/data/公开数据集清单.md、data/DATA_MANIFEST.csv、data/self_collected_prelabels.csv和coordination/CURRENT_ROUND.md。核验每个数据源的任务、规模、标注、许可证、下载状态和同源重复，评估190张自采图适合分类、检测、分割还是多任务。明确哪些视觉标签可直接复核，哪些病名必须由农技专家或实验检测确认。设计按拍摄者、连拍组和树/地块分组的无泄漏评测协议。只修改reports/2026-09-05/data-label-audit.md和research/inbox/dataset-audit.csv，完成后运行校验并把结果写入报告。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`，并附任务—标签—成本矩阵。

#### Report Result

Waiting.

### 方法方案Agent

Dispatch Status: blocked

#### Work Content

在两个上游报告完成后，形成分类、检测、分割三套互相独立的候选方案并排序。

#### Modification Scope

```text
reports/2026-09-05/method-options.md
```

#### Forbidden Scope

```text
上游报告完成前不得启动
src/
完整数据与权重
```

#### Dependency Documents

```text
reports/2026-09-05/cross-crop-methods.md
reports/2026-09-05/data-label-audit.md
coordination/PROJECT_STATE.md
coordination/DECISIONS.md
本任务表
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
rg -n "研究问题|标签需求|基线|消融|风险|兜底" reports/2026-09-05/method-options.md
```

#### Dispatch Prompt

```text
BLOCKED：等待跨作物文献报告和数据标签审计报告完成。完成后再根据两个报告形成分类、检测、分割三套候选方案，逐项说明研究问题、数据与标签、模型、创新、基线、消融、算力、风险和兜底，并给出排序。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`，并附三方案对比表和最终推荐条件。

#### Report Result

Waiting.

### 实验Agent

Dispatch Status: blocked

#### Work Content

等待方法方案和标签协议稳定后建立先导实验。

#### Modification Scope

```text
src/
configs/
reports/<实验日期>/
```

#### Forbidden Scope

```text
方法方案未验收前不得启动大规模训练
不得提交权重、数据集或runs目录
```

#### Dependency Documents

```text
已验收的方法报告
已验收的数据标签报告
coordination/PROJECT_STATE.md
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
```

#### Dispatch Prompt

```text
BLOCKED：等待协调者确认方法方案和标签协议。不要提前启动训练。
```

#### Required Report

实验环境、数据划分、命令、配置、种子、指标、失败项、产物路径和SHA-256。

#### Report Result

Waiting.

## Modification Paths

```text
reports/YYYY-MM-DD/
research/inbox/
src/与configs/仅在实验Agent解锁后
```

## Validation Commands

```powershell
python scripts/validate_repository.py
git status --short
```

## Risk Points

1. 多Agent同时编辑中央文档会造成冲突，普通Agent只能写独立报告。
2. 大文件、权重或原图误提交会破坏仓库轻量性和隐私边界。
3. 自采图预标签未经专家确认，不能直接作为病名真值。
4. 同叶、同果和同场景连续照片可能造成评测泄漏。

## Current Next Instruction

跨作物文献Agent与数据标签Agent可以并行执行各自完整Dispatch Prompt。

## Next Instructions

同上。两个上游报告通过验收后，由协调者解锁方法方案Agent。

## Coordinator Conclusion

本轮处于调研并行阶段。文献Agent和数据标签Agent为ready；方法Agent和实验Agent为blocked。

