# 2026-11-24：类条件复核与掩膜协议轮

> 上一轮（2026-11-22 方向 A 验证轮）已被 11-23/24 的证据改写：分类一致性消融按预注册判据停止，位置对照可行性与病原学评估给出新的优先级。历史见 `coordination/HISTORY_INDEX.md`。

## Current Judgment

1. 类条件风险排序是目前唯一有 2.5–4.6 倍增益证据的方向，且约 100 条 PCR 校准样本即可。
2. 按置信度/分歧送审在低预算下不如随机，不再做。
3. 上下文/位置方向缺掩膜与接缝匹配，暂不做第二轮干预实验。
4. 公司照片 0 张、专家盲评 0 份回填，目标域证据仍空。

## Round Goal

1. 把“类条件复核”写成可预注册的正式方案（含校准集与目标域同分布要求）。
2. 写一份前景掩膜与接缝匹配协议，供后续上下文/位置实验使用。
3. 推动专家盲评回填与公司照片获取（外部条件）。

## Gate Conditions

1. 方案未预注册前不跑新实验。
2. 校准集必须与目标域同分布，且不得用评估集自身做校准。
3. 掩膜协议必须先解决“只改背景”的验证方式，再谈位置效应。
4. 任何训练性任务保持 blocked。

## Global Forbidden Scope

```text
完整数据集、自采原图和模型权重
coordination/PROJECT_STATE.md（普通Agent只读）
coordination/DECISIONS.md（普通Agent只读）
coordination/USER_REQUIREMENTS.md（普通Agent只读）
其他Agent的报告文件
在方案预注册前启动训练或扩大干预实验
```

## Agent Division

### 类条件复核方案Agent

Dispatch Status: ready

对应GitHub Issue：待创建

#### Work Content

把“按预测类别的历史病原学错误率排序送审”写成正式方案：定义校准集、排序分数、预算、评价指标与判定阈值；明确校准集与目标域同分布的要求，以及 100 条样本的覆盖类别。

#### Modification Scope

```text
reports/2026-11-24/class-conditional-review-plan.md
research/inbox/
```

#### Forbidden Scope

```text
不得训练模型
不得用评估集自身做校准
不得改已验证的 597 样本口径
```

#### Dependency Documents

```text
README.md
coordination/USER_REQUIREMENTS.md
docs/项目现状_20261124.md
docs/experiments/病原学层面评估_20261123.md
coordination/PROJECT_STATE.md
本任务表
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
rg -n "校准集|同分布|预算|判定|覆盖" reports/2026-11-24/class-conditional-review-plan.md
```

#### Dispatch Prompt

```text
你是本轮类条件复核方案Agent。依次阅读README.md、AGENTS.md、coordination/USER_REQUIREMENTS.md、docs/项目现状_20261124.md、docs/experiments/病原学层面评估_20261123.md、coordination/PROJECT_STATE.md和本任务表。把“按预测类别历史病原学错误率排序送审”写成可预注册的正式方案：校准集怎么来、排序分数怎么算、预算档位、评价指标、判定阈值，以及校准集与目标域必须同分布的理由。明确约100条样本应覆盖哪些类别、如何避免用评估集自身做校准。只修改reports/2026-11-24/class-conditional-review-plan.md和research/inbox/，不训练、不调参，完成后运行校验并把结果写入报告。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`，附方案伪代码、预算表与风险清单。

#### Report Result

Waiting.

### 掩膜与接缝协议Agent

Dispatch Status: ready

对应GitHub Issue：待创建

#### Work Content

设计一份可执行的“前景掩膜 + 接缝匹配”协议，回答：如何获得可信叶片掩膜、如何验证只改了背景、如何在近端/远端等面积替换时匹配接缝强度。

#### Modification Scope

```text
reports/2026-11-24/mask-seam-protocol.md
```

#### Forbidden Scope

```text
不得用自动绿/边阈值冒充掩膜
不得在协议未冻结前跑扩散或生成模型
```

#### Dependency Documents

```text
docs/experiments/等面积位置先导_20261123.md
docs/experiments/上下文干预先导_20261123.md
docs/项目现状_20261124.md
本任务表
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
rg -n "掩膜|接缝|只改背景|验证|样本量" reports/2026-11-24/mask-seam-protocol.md
```

#### Dispatch Prompt

```text
你是本轮的掩膜与接缝协议Agent。依次阅读README.md、AGENTS.md、docs/experiments/等面积位置先导_20261123.md、docs/experiments/上下文干预先导_20261123.md、docs/项目现状_20261124.md和本任务表。设计可执行协议：可信叶片掩膜从哪来（人工/分割/半自动）及质控指标；如何证明“只改背景”；近端远端等面积替换如何匹配接缝强度；需要多少样本才能区分位置与替换量。不要用自动绿/边阈值冒充掩膜，不跑生成模型。只修改reports/2026-11-24/mask-seam-protocol.md，完成后运行校验并把结果写入报告。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`，附质控指标与失败案例处理规则。

#### Report Result

Waiting.

### 专家盲评与数据获取Agent

Dispatch Status: blocked

#### Work Content

推动 16 组盲评包回填与公司照片/许可获取。

#### Modification Scope

```text
reports/2026-11-24/external-data-status.md
```

#### Forbidden Scope

```text
不得把未回填的盲评当已复核
不得提交原始图片
```

#### Dependency Documents

```text
docs/data/真实照片场景核查_20261124.md
coordination/USER_REQUIREMENTS.md
本任务表
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
```

#### Dispatch Prompt

```text
BLOCKED：需要外部条件（专家时间、合作方提供照片与许可）。当前只记录进度，不自行扩展。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`。

#### Report Result

Waiting.

### 干预实验第二阶段（如启动）

Dispatch Status: blocked

#### Work Content

若掩膜协议完成且类条件方案稳定，再考虑重新预注册第二阶段。

#### Modification Scope

```text
src/
configs/
reports/<实验日期>/
```

#### Forbidden Scope

```text
不得沿用被判据 C 否决的旧协议
不得同时占用 GPU 与评价队列
```

#### Dependency Documents

```text
reports/2026-11-24/mask-seam-protocol.md
reports/2026-11-24/class-conditional-review-plan.md
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
```

#### Dispatch Prompt

```text
BLOCKED：需要新的预注册判据与掩膜协议。旧判据 C（翻转集重叠<50%）已被实测否决，不得原样复用。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`，附新判据与停止条件。

#### Report Result

Waiting.

## Modification Paths

```text
reports/2026-11-24/
research/inbox/
src/与configs/仅在四号任务解锁后
```

## Validation Commands

```powershell
python scripts/validate_repository.py
git status --short
```

## Risk Points

1. 类先验仍是同一数据集内交叉拟合，换到目标域可能失效。
2. 掩膜若靠自动分割，覆盖率不稳，可能把病灶本身切掉。
3. 接缝匹配只能匹配灰度，颜色/纹理差异仍可能残留。
4. 盲评未回填前，任何“场景导致误判”的说法都缺证据。
5. 旧判据被否后容易“换个判据再试”，必须记录新判据的事前理由。

## Current Next Instruction

并行执行“类条件复核方案Agent”与“掩膜与接缝协议Agent”。外部数据与第二阶段保持 blocked。

## Next Instructions

两份方案通过验收后，由协调者决定先做哪一个的正式实验；不得同时启动。

## Coordinator Conclusion

本轮为方案轮：两个 ready（类条件复核、掩膜接缝），外部数据与第二阶段 blocked；所有新实验须重新预注册。
