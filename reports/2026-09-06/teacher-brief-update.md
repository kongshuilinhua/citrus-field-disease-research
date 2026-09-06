# 阶段性研究思路汇报更新记录

## 任务

更新可直接转发的阶段性研究思路汇报，删除阅读对象和内部用途等元说明，并纳入最新方案复核中已经确认的有效结论。

## 实际完成

- 删除正文开头“向老师汇报用”等内部说明。
- 将“希望老师帮助确认”改为客观的“当前需要确认的问题”。
- 根据190张试拍统计，取消“检测默认主线”，增加分类＋弱定位与检测的先导决策门。
- 将多源柑橘迁移与背景鲁棒学习调整为当前优先创新。
- 将分类—定位解耦改为检测成立后的候选创新。
- 将症状到病名的单层/树状标签改为器官、症状、诊断和证据四轴标签。
- 统一类别数量、先导实验、模型比较、流程图、消融和下一步安排。
- 在用户诉求中增加“阅读对象只影响语气，不写入可转发正文”的长期规则。

## 证据与来源

- 自采图结构与复核负担：`data/self_collected_prelabels.csv`。
- Kimi方案评价与独立复核：`reports/2026-09-06/plan-feasibility-review.md`、`reports/2026-09-06/kimi-review-supplement.md`。
- 当前项目约束：`coordination/USER_REQUIREMENTS.md`、`coordination/PROJECT_STATE.md`。

## 修改文件

- `docs/research/阶段性研究思路汇报_20260905.md`
- `coordination/USER_REQUIREMENTS.md`
- `reports/2026-09-06/teacher-brief-update.md`

## 验证命令与结果

```powershell
rg -n "老师|导师|给.*看|本文|Kimi|Agent" docs/research/阶段性研究思路汇报_20260905.md
python scripts/validate_repository.py
git diff --check
```

验证结果：正文未检出“老师、导师、给某人看、本文、Kimi、Agent”等面向对象或协作元话语；仓库校验通过；`git diff --check` 无输出。

## 未完成或阻塞

- 研究任务仍需由200～300张正式先导数据决定，这是研究状态，不再以内部说明形式写入正文。
- 数据与标签完整审计尚未完成。

## 对下游的影响

- 后续Agent修改任何可直接转发文档时，必须删除阅读对象、生成过程和协作过程等元话语。
- 方法方案不得把检测或分类预设为最终主线。

## 下一步建议

在果园发生清单和App采集要求明确后，再更新类别列表和最终题目，不提前增加未经数据支持的方法模块。
