# 2026-11-19：现状同步与"结果到主张"判读

> 上一轮（2026-09-05 跨作物与数据可行性调研）已完成并被本轮取代，历史见 `coordination/HISTORY_INDEX.md`。
> 本轮不再启动新的训练性方法实验，先把已有结果判读清楚。

## Current Judgment

1. 实际主线已是复杂田间检测 + 可靠性；分类不再是主攻方向，但用户口径尚未正式改写。
2. SciDB 三条件基线与七架构矩阵已稳定，方法改进至今全部阴性。
3. testB/testC 是同一批对象的历史探索集，不能当独立验证。
4. 胡柚真值与设备配对数据仍缺，不能报告胡柚准确率。

## Round Goal

1. 对已有实验数字做独立的"结果到主张"判读，列出每条主张的支持程度与缺失证据。
2. 核查 `DeepSeek分析_20261110/` 等既有目录的预算、数据可见性与结论边界，确定哪些结果可直接复用、哪些必须重跑。
3. 明确下一阶段唯一值得投入的假设，并写出停止条件。

## Gate Conditions

1. 两个判读任务可并行，只读现有产物，不训练、不改历史结果。
2. 任何训练性实验必须等判读结论并经协调者确认后才解锁。
3. 胡柚相关任务全部等专家标注或配对采样，当前为 blocked。

## Global Forbidden Scope

```text
完整数据集、自采原图和模型权重
coordination/PROJECT_STATE.md（普通Agent只读）
coordination/DECISIONS.md（普通Agent只读）
coordination/USER_REQUIREMENTS.md（普通Agent只读）
其他Agent的报告文件
训练、评价与 GPU 任务
```

## Agent Division

### 结果到主张判读Agent

Dispatch Status: ready

对应GitHub Issue：待创建

#### Work Content

读取状态快照、检测矩阵、可靠性结论、长期目标各轮记录与最新先导，逐条判断"这个数字能支撑什么主张、不能支撑什么主张"，特别标注被重复使用过的探索集结论和不成立的外推。

#### Modification Scope

```text
reports/2026-11-19/result-to-claim.md
research/inbox/claim-evidence.csv
```

#### Forbidden Scope

```text
coordination/
src/
任何训练或评价命令
```

#### Dependency Documents

```text
README.md
coordination/USER_REQUIREMENTS.md
coordination/HISTORY_INDEX.md
docs/项目现状_20261119.md
coordination/PROJECT_STATE.md
coordination/DECISIONS.md
本任务表
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
rg -n "支持|不支持|缺失|外推|探索集" reports/2026-11-19/result-to-claim.md
```

#### Dispatch Prompt

```text
你是本轮结果到主张判读Agent。依次阅读README.md、AGENTS.md、coordination/USER_REQUIREMENTS.md、coordination/HISTORY_INDEX.md、docs/项目现状_20261119.md、coordination/PROJECT_STATE.md、coordination/DECISIONS.md和本任务表。只根据仓库与本地证据文件中已有的数字，逐条判断哪些主张被支持、哪些部分支持、哪些不被支持，并列出每条主张缺失的证据。特别检查：testB/testC是否被当作独立验证、源val与外部条件方向相反的结论是否被过度外推、种子标准差不支持"某架构最好"的说法。只修改reports/2026-11-19/result-to-claim.md和research/inbox/claim-evidence.csv，不运行任何训练或评价命令，完成后运行校验并把结果写入报告。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`，并附"主张—证据—支持程度"三列表。

#### Report Result

Waiting.

### 既有结果复用核查Agent

Dispatch Status: ready

对应GitHub Issue：待创建

#### Work Content

核查 `DeepSeek分析_20261110/`、`方法路线扩展_20261031/` 等目录中已存在的结论：确认其数据可见性（是否访问过 B/C）、预算（种子数、轮数、选参范围）、复算方式，给出"可直接引用 / 需补算 / 不可用"的判定。

#### Modification Scope

```text
reports/2026-11-19/evidence-reuse-audit.md
research/inbox/evidence-reuse.csv
```

#### Forbidden Scope

```text
coordination/
src/
任何训练或评价命令
```

#### Dependency Documents

```text
README.md
coordination/USER_REQUIREMENTS.md
docs/项目现状_20261119.md
coordination/PROJECT_STATE.md
本任务表
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
rg -n "可直接引用|需补算|不可用|预算|种子|B/C" reports/2026-11-19/evidence-reuse-audit.md
```

#### Dispatch Prompt

```text
你是本轮既有结果复用核查Agent。依次阅读README.md、AGENTS.md、coordination/USER_REQUIREMENTS.md、docs/项目现状_20261119.md、coordination/PROJECT_STATE.md和本任务表。逐个核查本地已有结论目录，确认三点：一是是否在选参或调参中访问过B/C，二是种子数、轮数与停止条件，三是有没有独立同进程FP32复算与保存预测。给出"可直接引用/需补算/不可用"判定和理由。只修改reports/2026-11-19/evidence-reuse-audit.md和research/inbox/evidence-reuse.csv，不运行训练或评价，完成后运行校验并把结果写入报告。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`，并附"目录—判定—理由"表。

#### Report Result

Waiting.

### 训练性方法实验Agent

Dispatch Status: blocked

#### Work Content

在判读结论确认后，按冻结协议实现下一阶段唯一假设。

#### Modification Scope

```text
src/
configs/
reports/<实验日期>/
```

#### Forbidden Scope

```text
判读报告未验收前不得启动任何训练
不得提交权重、数据集或runs目录
不得同时占用GPU与评价队列
```

#### Dependency Documents

```text
reports/2026-11-19/result-to-claim.md
reports/2026-11-19/evidence-reuse-audit.md
coordination/PROJECT_STATE.md
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
```

#### Dispatch Prompt

```text
BLOCKED：等待两份判读报告通过验收。启动前必须说明要修的具体错误、对照条件、种子数与停止条件，经协调者确认。
```

#### Required Report

实验环境、数据划分、命令、配置、种子、指标、失败项、产物路径和SHA-256。

#### Report Result

Waiting.

### 胡柚真值与配对数据Agent

Dispatch Status: blocked

#### Work Content

获取专家标注与同对象/同叶配对样本。

#### Modification Scope

```text
reports/2026-11-19/field-data-status.md
```

#### Forbidden Scope

```text
不得把视觉预标签写成病原学确诊
不得提交原始图片
```

#### Dependency Documents

```text
docs/项目现状_20261119.md
data/DATA_MANIFEST.csv
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
```

#### Dispatch Prompt

```text
BLOCKED：需要外部条件（专家时间、同叶两台手机重新拍摄）。当前只记录缺口，不启动。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`。

#### Report Result

Waiting.

## Modification Paths

```text
reports/2026-11-19/
research/inbox/
src/与configs/仅在三号任务解锁后
```

## Validation Commands

```powershell
python scripts/validate_repository.py
git status --short
```

## Risk Points

1. 判读任务容易变成"再写一遍总结"，必须逐条给支持程度，不能只复述。
2. 既有目录数量多、口径杂，复用前必须核是否碰过 B/C，否则会把调参结果当独立验证。
3. B/C 已被多次使用，任何新结论都只能算探索性，不能称为泛化证据。
4. 未拿到专家真值前，胡柚相关数字一律不报。

## Current Next Instruction

并行执行"结果到主张判读Agent"与"既有结果复用核查Agent"的 Dispatch Prompt。训练性实验与胡柚任务保持 blocked。

## Next Instructions

判读报告通过验收后，由协调者选择唯一假设并解锁实验Agent。

## Coordinator Conclusion

本轮为状态同步与判读轮：仓库内容已更新到 2026-11-19 现状，两个只读判读任务为 ready，训练与胡柚任务 blocked。
