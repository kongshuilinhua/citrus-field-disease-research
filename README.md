# 胡柚复杂田间病虫害视觉研究

这是一个用于多地点、多 Agent 协作的私有研究仓库。仓库保存研究状态、文献结论、数据清单、实验代码、配置和轻量结果，不保存完整数据集、模型权重或受版权限制的论文 PDF。

## 新 Agent 快速开始

按以下顺序阅读：

1. `AGENTS.md`：协作边界和强制报告规则。
2. `coordination/USER_REQUIREMENTS.md`：用户当前有效诉求和偏好。
3. `coordination/HISTORY_INDEX.md`：项目时间线和历史证据入口。
4. `docs/项目接手文档.md`：研究背景、已有实验、数据和风险。
5. `coordination/DECISIONS.md`：已经确认的决策。
6. `coordination/PROJECT_STATE.md`：当前唯一正式状态。
7. `coordination/CURRENT_ROUND.md`：当前任务、修改范围和可复制提示词。

完成任务后，必须在 `reports/YYYY-MM-DD/` 下新增独立报告，并通过仓库校验。Agent 之间不会自动通信；Git、Issue、PR和报告文件是唯一协作媒介。

## 仓库保存什么

- `docs/`：交接、调研、数据和实验结论文档。
- `coordination/`：中央状态、决策和当前轮次任务。
- `reports/`：每个 Agent 的独立、可追踪报告。
- `research/`：文献索引及待合并研究条目。
- `data/`：数据清单、轻量预标注和标签规范。
- `models/`：模型权重清单，不含权重文件。
- `src/`：基线和论文复现代码。
- `configs/`：不含本机秘密的示例配置。
- `artifacts/summaries/`：小型 JSON/CSV 指标摘要。

## 大文件策略

完整数据、权重和训练目录保存在外部存储。仓库仅记录来源、许可证、路径、大小、SHA-256和下载状态。禁止提交：

- `*.pt`、`*.pth`、`*.ckpt`、`*.onnx`、`*.engine`；
- 完整图片目录、数据集压缩包和训练输出；
- 未确认再分发权限的论文 PDF；
- 包含个人隐私或未经许可公开的自采原图。

## 本地校验

```powershell
python scripts/validate_repository.py
```

成功标准：没有禁用扩展名、没有超过20 MiB的文件、关键协作文档存在、所有报告包含规定章节。

## 当前研究判断

纯净背景分类已经接近饱和，真实问题集中在复杂田间背景、跨品种/跨拍摄者泛化、相似症状、少样本和病名可靠性。YOLO既不预设为唯一主线，也不被排除；应与分类、Faster R-CNN、DETR/RT-DETR、分割和多任务方案按同一协议比较。

本仓库暂不提供公开许可证。未经项目负责人确认，不得公开自采数据或重新分发第三方数据和论文。
