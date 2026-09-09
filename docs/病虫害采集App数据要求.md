# 胡柚、香柚病虫害采集App数据要求

## 一、基本规则

- 一个独立采集对象（叶片、果实、枝条或虫体）对应一个 `sample_id`，每张照片对应一个 `image_id`。
- 同一对象的近景图和稍远图共用 `sample_id`，但使用不同的 `image_id`。
- 不同对象不得共用 `sample_id`。
- 类别列表应支持后台配置，不能写死在App中。
- “不确定”必须可以正常提交，不能强制选择具体病虫害。
- 拍摄人员不需要画检测框或病斑掩膜。

建议约20%～30%的对象同时拍摄近景和稍远图，其余对象可以只拍一张清晰近景。

## 二、数据字段

| 字段 | 要求 | 说明 |
|---|---|---|
| `schema_version` | 必填、自动记录 | 数据结构版本，如 `1.0` |
| `sample_id` | 必填、自动生成 | 同一采集对象保持一致 |
| `image_id` | 必填、自动生成 | 每张图片唯一 |
| `cultivar` | 必填 | `huyou` / `xiangyou` / `unknown` |
| `organ_type` | 必填 | `leaf` / `fruit` / `branch` / `insect` / `other` |
| `suspected_label` | 必填 | 初步类别编码；允许 `unknown` |
| `symptom_tags` | 可选、多选 | 黄化、斑驳、坏死、潜道、煤污、木栓化等 |
| `orchard_id` | 必填 | 匿名果园或地块编号，不填写具体地址 |
| `tree_id` | 建议 | 能够区分是否来自同一棵树即可 |
| `collector_id` | 必填 | 匿名采集人员编号 |
| `collection_batch_id` | 必填、自动生成 | 区分采集日期或任务批次 |
| `view_type` | 必填 | `close_up` / `context` / `other` |
| `captured_at` | 必填、自动记录 | 带时区的实际拍摄时间 |
| `device_model` | 自动记录 | 手机或相机型号 |
| `file_name` | 自动记录 | 建议使用 `image_id.jpg` |
| `width`、`height` | 自动记录 | 原图像素尺寸 |
| `mime_type` | 自动记录 | 优先使用 `image/jpeg` |
| `file_size` | 自动记录 | 文件字节数 |
| `sha256` | 上传后生成 | 文件完整性检查和精确去重 |
| `upload_status` | 自动维护 | `pending` / `uploading` / `success` / `failed` |
| `taxonomy_version` | 必填 | 类别表版本，如 `citrus_v1` |
| `review_status` | 必填 | `unreviewed` / `confirmed` / `corrected` / `unresolved` |
| `reviewed_label` | 条件填写 | 类别被确认或修改后填写 |
| `note` | 可选 | 补充说明或无法判断的原因 |

类别应使用稳定编码保存，界面显示中文名称。类别名称调整时发布新的 `taxonomy_version`，不能直接改变历史编码的含义。

## 三、图片与上传要求

1. 保存原始RGB图片，不添加水印、文字、滤镜、检测框或人工标记。
2. 服务端可以生成缩略图，但不能覆盖原图。
3. 建议原图长边不低于1600像素；严重模糊、过曝或主体不可见时提示重新拍摄。
4. 网络不可用时允许离线暂存，恢复网络后继续上传。
5. 上传失败必须支持重试，且不能生成新的 `sample_id` 或重复图片记录。
6. 服务端以 `image_id` 作为幂等键；重复上传同一 `image_id` 时不得新增记录。
7. 只有图片和元数据都保存成功后，才显示提交成功。
8. 未上传成功的数据不得自动删除。

黄化、斑驳等图片在没有可靠检测依据时，只能保存为可见症状或“待确认”，不能保存为“黄龙病确诊”。

## 四、导出要求

后台需要能够按日期或采集批次导出：

```text
export/
├─ images/          # 原始图片
├─ samples.csv      # 对象和采集信息
├─ images.csv       # 图片信息
├─ reviews.csv      # 复核记录
└─ taxonomy.json    # 类别编码、中文名称和版本
```

- CSV使用UTF-8编码，时间使用ISO 8601格式并保留时区。
- 文件名及表格中的 `sample_id`、`image_id` 必须准确对应。
- 不要在导出时随机划分训练集、验证集和测试集。
- 果园、树木和采集人员使用匿名编号；精确定位不是必填字段。
- 普通采集人员不能覆盖已经确认的复核结果。

## 五、验收检查

1. 同一对象可以上传近景和稍远图，并保持相同 `sample_id`。
2. 断网拍摄后能够恢复上传，且不会产生重复数据。
3. “不确定”类别能够正常提交。
4. 原图、缩略图和元数据能够正确对应。
5. 导出的CSV无中文乱码，图片和记录数量一致。
6. 类别表升级后，历史数据仍能还原原类别含义。
7. 普通采集人员不能修改已经确认的复核结果。

## 六、数据示例

```json
{
  "schema_version": "1.0",
  "sample_id": "d879afc0-63d9-4f86-9e29-e7a5c646d314",
  "image_id": "b31f4686-51e0-4eaa-8e89-e56525dc6755",
  "cultivar": "huyou",
  "organ_type": "leaf",
  "suspected_label": "unknown",
  "symptom_tags": ["mottling", "yellowing"],
  "orchard_id": "orchard_01",
  "tree_id": "tree_023",
  "collector_id": "collector_07",
  "collection_batch_id": "2026-09-orchard01-a",
  "view_type": "close_up",
  "captured_at": "2026-09-09T10:35:20+08:00",
  "device_model": "phone_model",
  "file_name": "b31f4686-51e0-4eaa-8e89-e56525dc6755.jpg",
  "width": 3072,
  "height": 4096,
  "mime_type": "image/jpeg",
  "file_size": 2851043,
  "sha256": "8f0d8d52a203c6f1dfc342298b6d4b84653cb2609b77d3fbd4c972175e28f458",
  "upload_status": "success",
  "taxonomy_version": "citrus_v1",
  "review_status": "unreviewed"
}
```
