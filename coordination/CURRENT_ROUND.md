# 2026-11-22：方向 A 验证轮（基线复现与预注册对照）

> 上一轮（2026-11-19 状态同步与判读）已并入 `docs/项目现状_20261122.md`，历史见 `coordination/HISTORY_INDEX.md`。
> 本轮目标只有一个：**验证“多粒度一致性”是不是独立贡献，还是只是多尺度训练**。

## Current Judgment

1. 换模型与训练技巧已证明天花板很低（最大 +0.93，且单种子），不再作为主线。
2. 跨粒度失效（HLB 跨域 0%，整体 −23.7 点）可复现，是当前最有实证支撑的选题。
3. 高置信度误分类只占 4%，不构成方法动机。
4. 数据侧有两处必须先修：testA 9 张同图、随机划分未按同叶/连拍分组。

## Round Goal

1. 复现并冻结 B1–B4 四个基线。
2. 实现“多粒度一致性”，与 B3 做预注册对照。
3. 修正 testA 泄漏，补 group split。
4. 查清跨粒度一致性是否已有先例。

## Gate Conditions

1. B1–B4 未跑完，不得实现新方法。
2. 对照结论只看“是否显著优于 B3”，不看是否优于 B1/B2。
3. 单种子结果一律标“初步”，不得写入正式结论。
4. 数据修正与方法实验可以并行，但不得同时占用 GPU。

## Global Forbidden Scope

```text
完整数据集、自采原图和模型权重
coordination/PROJECT_STATE.md（普通Agent只读）
coordination/DECISIONS.md（普通Agent只读）
coordination/USER_REQUIREMENTS.md（普通Agent只读）
其他Agent的报告文件
在 B1–B4 冻结前新增其他改进模块
```

## Agent Division

### 基线复现Agent（B1–B4）

Dispatch Status: ready

对应GitHub Issue：待创建

#### Work Content

复现四个基线并冻结：B1 只用裁框、B2 只用整图、B3 裁框 + 多尺度训练、B4 裁框 + 整图特征拼接。统一数据、统一训练轮数、统一选参协议，各跑至少 3 个种子。

#### Modification Scope

```text
src/
configs/
reports/2026-11-22/baselines-b1-b4.md
```

#### Forbidden Scope

```text
在四个基线冻结前实现“多粒度一致性”
用 B/C 选参
```

#### Dependency Documents

```text
README.md
coordination/USER_REQUIREMENTS.md
docs/项目现状_20261122.md
coordination/PROJECT_STATE.md
本任务表
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
rg -n "种子|轮数|选参|B1|B2|B3|B4" reports/2026-11-22/baselines-b1-b4.md
```

#### Dispatch Prompt

```text
你是本轮基线复现Agent。依次阅读README.md、AGENTS.md、coordination/USER_REQUIREMENTS.md、docs/项目现状_20261122.md、coordination/PROJECT_STATE.md和本任务表。复现B1（只用裁框）、B2（只用整图）、B3（裁框+多尺度训练）、B4（裁框+整图特征拼接）四个基线，统一数据与训练轮数，仅用源val选参，各跑至少3个种子并逐种子报告。B3是与候选方法对照的关键基线，必须与B4用同一预算。只修改src/、configs/和reports/2026-11-22/baselines-b1-b4.md，完成后运行校验并把结果写入报告。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`，附四基线逐种子表与选参记录。

#### Report Result

Waiting.

### 数据修正Agent

Dispatch Status: ready

对应GitHub Issue：待创建

#### Work Content

1. 剔除 testA 与训练集同图的 9 张，用现有权重复算 mAP50/50–95，给出披露前后对照。
2. 按 dHash 近重复对全库分组，产出按“同叶/连拍组”整组划分的 split 方案。
3. 按原始 `label_object.json` 复核 S12 是否存在于 testB/testC，澄清 `docs/data/数据集可用性复核_20261122.md` 与其他文档的冲突。

#### Modification Scope

```text
docs/data/
reports/2026-11-22/data-fix.md
research/inbox/
```

#### Forbidden Scope

```text
不得删除或覆盖原始数据
不得在没有记录的情况下替换现有划分
```

#### Dependency Documents

```text
docs/data/数据集可用性复核_20261122.md
docs/data/scidb复核_20261122/
coordination/PROJECT_STATE.md
本任务表
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
rg -n "剔除|复算|分组|S12|label_object" reports/2026-11-22/data-fix.md
```

#### Dispatch Prompt

```text
你是本轮数据修正Agent。依次阅读README.md、AGENTS.md、docs/data/数据集可用性复核_20261122.md、docs/data/scidb复核_20261122/下的脚本与JSON、coordination/PROJECT_STATE.md和本任务表。完成三件事：剔除testA与训练集同图的9张后复算mAP50与50-95并给对照；用现有dHash近重复结果产出按同叶/连拍组的group split；按原始label_object.json复核S12是否存在于testB/testC，明确写出结论与证据。不得删除原始数据。只修改docs/data/、report文件和research/inbox/，完成后运行校验并把结果写入报告。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`，附剔除前后对照表与分组统计。

#### Report Result

Waiting.

### 文献核查Agent

Dispatch Status: ready

对应GitHub Issue：待创建

#### Work Content

核查“cross-granularity consistency / 跨粒度一致性”在植物病害与通用视觉中是否已有工作，给出可引用的最近邻与方法差异。

#### Modification Scope

```text
reports/2026-11-22/literature-cross-granularity.md
research/inbox/cross-granularity-evidence.csv
```

#### Forbidden Scope

```text
不得用搜索摘要当证据
不得把“多尺度训练”已有的结论当作新发现
```

#### Dependency Documents

```text
docs/项目现状_20261122.md
reports/2026-09-05/method-landscape-expanded.md
本任务表
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
rg -n "DOI|代码|粒度|一致性|排除" reports/2026-11-22/literature-cross-granularity.md
```

#### Dispatch Prompt

```text
你是本轮文献核查Agent。依次阅读README.md、AGENTS.md、docs/项目现状_20261122.md和本任务表。核查“跨粒度一致性”（同一病害在整图与局部裁框上的表征一致性）是否已有工作，重点看植物病害、细粒度识别、多尺度一致性、域泛化四类文献。只使用一手论文、官方代码和正式页面，每篇记录任务、粒度定义、方法、对照、代码可得性与排除理由。只修改reports/2026-11-22/literature-cross-granularity.md和research/inbox/cross-granularity-evidence.csv，完成后运行校验并把结果写入报告。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`，附最近邻对比表与“是否已有先例”的明确结论。

#### Report Result

Waiting.

### 方法实现Agent（多粒度一致性）

Dispatch Status: blocked

#### Work Content

在 B1–B4 冻结后实现“多粒度一致性”，与 B3 做预注册对照。

#### Modification Scope

```text
src/
configs/
reports/2026-11-23/
```

#### Forbidden Scope

```text
B1–B4 未冻结前不得启动
不得在对照不利时更换指标或追加模块
```

#### Dependency Documents

```text
reports/2026-11-22/baselines-b1-b4.md
coordination/PROJECT_STATE.md
本任务表
```

#### Validation Commands

```powershell
python scripts/validate_repository.py
```

#### Dispatch Prompt

```text
BLOCKED：等待B1–B4四个基线冻结。启动前必须先写出预注册的判定标准：只有在多种子下显著优于B3才算独立贡献；否则写负结果并停止该路线。
```

#### Required Report

按 `coordination/REPORT_TEMPLATE.md`，附与 B3 的逐种子对照与判定。

#### Report Result

Waiting.

## Modification Paths

```text
reports/2026-11-22/
research/inbox/
docs/data/
src/与configs/仅在对应任务解锁后
```

## Validation Commands

```powershell
python scripts/validate_repository.py
git status --short
```

## Risk Points

1. “多粒度一致性”很可能只是多尺度训练，B3 对照是唯一的证伪手段。
2. CitrusUAT 只有 443 张、5 类，方差大，5 折 CV 也不足以支撑强结论。
3. testA 同图未剔除前，任何 testA 数字都偏高。
4. 随机划分未按同叶分组，现有“泛化”结论都可能被同叶泄漏削弱。
5. S12 类别映射未核清前，引用任何 S12 数字都会踩到“张冠李戴”。

## Current Next Instruction

并行执行“基线复现Agent”“数据修正Agent”“文献核查Agent”三个 ready 任务。方法实现保持 blocked。

## Next Instructions

B1–B4 与文献结论出来后再决定是否启动方法实现；若文献显示已有先例，先与协调者确认创新边界。

## Coordinator Conclusion

本轮为方向 A 验证轮：三个只读/基线任务 ready，方法实现 blocked，判定标准已预注册。
