# 项目记忆与用户诉求入口补强报告

## 任务

核对私有协作仓库是否记录用户当前诉求，并确认其他Agent接手时是否会读取历史。补充明确的用户诉求文件和历史索引，避免依赖本机聊天或客户端私有memory。

## 实际完成

- 核对现有 `PROJECT_STATE.md` 已记录App、1000张、胡柚/香柚、病虫害范围、专家复核、黄龙病无PCR和方法不预设等最新条件。
- 确认原 `AGENTS.md` 已强制读取README、项目接手、中央状态、决策和当前任务。
- 新增 `coordination/USER_REQUIREMENTS.md`，集中记录用户当前有效诉求、偏好和希望避免的问题。
- 新增 `coordination/HISTORY_INDEX.md`，按时间整理调研、实验、自采数据、公开数据和GitHub协作演进。
- 将两个新文件加入README和AGENTS的强制阅读顺序。
- 更新仓库自动校验，将两个项目记忆文件设为必需文件。

## 证据与来源

- 用户诉求来源于当前项目对话和已验收的 `PROJECT_STATE.md`、`DECISIONS.md`、`research-direction-v1.md`、`method-landscape-expanded.md`。
- 历史实验数字来自 `docs/experiments/全部复现结果汇总.md`。
- 自采数据状态来自 `data/self_collected_prelabels.csv` 和项目接手文档。

## 修改文件

- `coordination/USER_REQUIREMENTS.md`
- `coordination/HISTORY_INDEX.md`
- `AGENTS.md`
- `README.md`
- `scripts/validate_repository.py`
- `reports/2026-09-05/project-memory-index.md`

## 验证命令与结果

```powershell
python scripts/validate_repository.py
rg -n "USER_REQUIREMENTS|HISTORY_INDEX|项目记忆|私有memory" README.md AGENTS.md
```

验证结果：通过。仓库必需文件、报告结构、文件大小和共享规则均符合当前约束；检索也确认README与AGENTS已纳入两个新增入口。

## 未完成或阻塞

- 仓库不能自动读取任何Agent客户端私有memory；只有提交到GitHub的文件才可跨设备共享。
- 新的用户要求仍需由协调者及时更新，否则历史索引可能过时。

## 对下游的影响

- 所有新Agent必须先读取用户诉求和历史索引，再读取详细接手文档和当前任务。
- 后续Agent不得声称依赖未提交的聊天记忆完成工作。

## 下一步建议

每次用户明确改变研究范围、数据条件、输出形式或协作规则时，由协调者同步更新用户诉求、中央状态和历史索引。
