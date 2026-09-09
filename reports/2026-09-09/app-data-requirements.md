# App数据要求文档编制记录

## 任务

单独编制一份病虫害采集App的数据要求，使开发人员能够据此实现字段、关联、上传、复核和导出功能。

## 实际完成

- 建立采集对象、图片和复核记录三类数据关系。
- 明确 `sample_id` 与 `image_id` 的生成和配对规则。
- 给出采集对象、图片、类别复核三组字段表。
- 规定类别表可配置、“不确定”可提交、黄龙病无可靠依据不作确诊。
- 规定原图保存、缩略图、离线暂存、失败重试和幂等上传要求。
- 规定CSV/JSON和原图导出结构、隐私权限及验收检查。
- 提供一份完整JSON数据示例。

## 证据与来源

- 用户确认的数据规模和App场景：`coordination/USER_REQUIREMENTS.md`。
- 图像采集要求：`docs/胡柚香柚病虫害图像数据采集规范.md`。
- 自采数据现有字段：`data/self_collected_prelabels.csv`。

## 修改文件

- 新增 `docs/病虫害采集App数据要求.md`。
- 更新 `coordination/USER_REQUIREMENTS.md` 和 `coordination/HISTORY_INDEX.md`。
- 新增 `reports/2026-09-09/app-data-requirements.md`。

## 验证命令与结果

```powershell
python scripts/validate_repository.py
rg -n "sample_id|image_id|unknown|upload_status|taxonomy_version|UTF-8|幂等|JSON" docs/病虫害采集App数据要求.md
git diff --check
```

验证结果：文档共115行；仓库校验通过；关键ID、`unknown`、类别版本、UTF-8导出、幂等上传和JSON示例均已检出；未检出阅读对象或内部协作元话语；`git diff --check` 无输出。

## 未完成或阻塞

- 最终后端接口路径、鉴权方式和数据库类型尚未确定，因此文档只规定数据契约，不绑定技术栈。
- 最终类别编码需要在果园病虫害清单确认后形成 `taxonomy_v1`。

## 对下游的影响

- App开发可以先按稳定字段和ID关系实施，不必等待最终模型确定。
- 后续模型训练和复核系统统一使用 `sample_id`、`image_id` 与 `taxonomy_version`。

## 下一步建议

开发开始前确认后端存储方式、账号权限、是否支持离线采集，以及类别表由谁维护。
