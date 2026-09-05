# 可迁移方法全景与高质量论文扩展调研

> 日期：2026-09-05  
> Agent/任务：跨作物方法扩展调研 / method-landscape-expanded  
> 对应Issue：#1  
> 分支：main，由协调者提交

## 任务

在不局限于柑橘、不局限于YOLO、也不预设分类/检测/分割的前提下，系统梳理可迁移到胡柚/香柚复杂果园近景病虫害识别的高质量方法。重点关注CVPR、ICCV、ICLR、NeurIPS等通用视觉顶会，以及Computers and Electronics in Agriculture、Biosystems Engineering、Engineering Applications of Artificial Intelligence、Plant Phenomics、Scientific Data等农业与工程期刊。

目标不是把所有方法堆入一个模型，而是建立方法池、适用条件、优先级和淘汰规则，为后续200—300张先导实验提供选择空间。

## 实际完成

1. 将候选方法扩展为十二个家族：两阶段检测、非YOLO一阶段检测、DETR、检测—分割统一、视觉基础模型、分类定位解耦、小目标高分辨率、半监督、跨域/域泛化、主动学习、开放世界、长尾/蒸馏/多模态。
2. 整理42篇高质量或强相关论文，记录任务、核心思想、代码、迁移方式、成本和优先级。
3. 给出“立即做基线”“有条件引入”“高风险储备”三层方法池。
4. 提炼六个可独立成题的研究方向，而不是只保留一个前景—背景方案。
5. 给出检测器筛选矩阵、分阶段实验顺序和停止条件。

## 证据与来源

完整结构化论文表见：

`research/inbox/method-landscape-evidence.csv`

本报告优先依赖论文原文、CVF/OpenReview/NeurIPS正式页面、期刊页面和官方代码仓库。以下按方法家族说明可迁移价值。

### 1. 两阶段高精度检测

代表方法：Faster R-CNN、Cascade R-CNN。

- Faster R-CNN先产生候选区域，再做分类与回归。它在小数据、少量前景目标和需要精细分类时通常比只追求速度的一阶段模型更值得比较。
- Cascade R-CNN使用逐级提高IoU阈值的检测头，减少低质量候选框和定位噪声，适合器官框边界质量要求较高的场景。
- 2026年棉花田间研究在独立数据上发现Faster R-CNN ResNet-101比YOLOv11更稳，说明农业任务不能默认YOLO占优。

迁移方式：

- 患病叶/果器官级bbox检测。
- 在RoI内部增加病斑局部特征分支。
- 使用MobileNetV3/ResNet50/Swin-T作为不同速度档骨干。

优点：训练稳定、RoI天然适合局部病征分类、小数据友好。  
缺点：速度较慢，手机端可能需要蒸馏或服务器部署。  
优先级：A。

关键论文：

- Faster R-CNN：https://arxiv.org/abs/1506.01497
- Cascade R-CNN：https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html
- 棉花田间Faster R-CNN vs YOLOv11：https://doi.org/10.1016/j.biosystemseng.2025.104379

### 2. 非YOLO一阶段检测

代表方法：RetinaNet、FCOS、GFL/ATSS类密集检测器。

- RetinaNet的Focal Loss专门处理大量简单背景负样本压倒少量前景的问题。
- FCOS采用逐像素、anchor-free检测，不需要为叶片和果实重新设计锚框。
- 对当前“复杂背景很多、受害器官较少”的数据，RetinaNet/FCOS能提供不同于YOLO的强一阶段对照。

迁移方式：

- 将健康背景和大量未受害叶片作为负样本。
- 使用FPN和高分辨率P2/P3特征检测较小受害器官。
- 与YOLO比较是否真正需要复杂颈部结构。

优点：代码成熟、算力可控、能隔离“YOLO家族特性”。  
缺点：仍依赖NMS；对相似病害的分类头未必足够。  
优先级：A-（至少选择FCOS或RetinaNet之一）。

关键论文：

- Focal Loss/RetinaNet：https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html
- FCOS：https://openaccess.thecvf.com/content_ICCV_2019/html/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.html

### 3. DETR与实时Transformer检测

代表方法：Deformable DETR、DINO、RT-DETR、Co-DETR、Cascade-DETR、Lite DETR。

- Deformable DETR以少量采样点处理多尺度特征，降低标准Transformer注意力开销。
- DINO通过去噪锚框、查询选择和改进框预测增强训练效率。
- RT-DETR是CVPR 2024实时端到端检测器，避免NMS，适合作为YOLO的直接实时对照。
- Co-DETR用一对多辅助头解决DETR正样本监督稀疏，但官方实现常依赖多GPU和较旧MMDetection，当前单卡复现成本较高。
- Cascade-DETR强调跨不同视觉域的通用检测和IoU感知置信度，概念上与跨果园泛化匹配。
- Lite DETR减少多尺度编码器开销，同时保留小目标所需的低层高分辨率信息。

迁移方式：

- RT-DETR-R18/R50作为第一批基线。
- 若RT-DETR在小目标或训练稳定性上不足，再测试DINO-R50/Deformable DETR。
- 不建议第一阶段直接上Co-DETR或ViT-L等重模型。

优点：全局上下文、端到端、对遮挡和多目标有潜力。  
缺点：小数据收敛、显存和训练周期可能不如Faster R-CNN/YOLO友好。  
优先级：A（RT-DETR）；B（DINO/Deformable）；C（Co-DETR）。

关键论文：

- Deformable DETR：https://arxiv.org/abs/2010.04159
- DINO：https://arxiv.org/abs/2203.03605
- RT-DETR：https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf
- Co-DETR：https://openaccess.thecvf.com/content/ICCV2023/html/Zong_DETRs_with_Collaborative_Hybrid_Assignments_Training_ICCV_2023_paper.html
- Cascade-DETR：https://openaccess.thecvf.com/content/ICCV2023/html/Ye_Cascade-DETR_Delving_into_High-Quality_Universal_Object_Detection_ICCV_2023_paper.html
- Lite DETR：https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Lite_DETR_An_Interleaved_Multi-Scale_Encoder_for_Efficient_DETR_CVPR_2023_paper.pdf

### 4. 分类与定位解耦

代表方法：TSD（Task-aware Spatial Disentanglement）、检测器双头解耦。

通用检测研究指出，适合分类的位置与适合框回归的位置并不相同：分类需要对象内部最有判别力的区域，定位需要对象边缘。胡柚场景中这种矛盾更明显：

- 器官定位依赖叶片/果实轮廓。
- 病虫害分类依赖器官内部的病斑、潜道、霉层或虫体。

这比“加注意力模块”更有明确问题依据。

可迁移设计：

1. 定位分支关注叶/果边界和整体形状。
2. 分类分支由DINOv2 patch、显著图或SAM前景内部采样病征区域。
3. 两分支只在决策层融合，避免定位边缘特征污染病害分类。

优点：与器官级bbox任务高度匹配，可形成清晰消融。  
缺点：若类别标签噪声高，局部病征分支会学到错误区域。  
优先级：A，值得作为独立主创新候选。

关键论文：

- TSD：https://openaccess.thecvf.com/content_CVPR_2020/html/Song_Revisiting_the_Sibling_Head_in_Object_Detector_CVPR_2020_paper.html

### 5. 小目标与高分辨率检测

代表方法：QueryDet、CFINet、AutoFocus、P2高分辨率分支。

如果标注对象是整片患病叶/果，目标通常不算特别小，这一方向优先级有限；如果后续改为病斑框、虫体框，才会成为核心。

- QueryDet先低分辨率定位可能含小目标的位置，再只在相关位置计算高分辨率特征。
- CFINet采用粗到细RPN和特征模仿增强小目标表征。
- AutoFocus只放大可能含小目标的区域，减少全图多尺度计算。

迁移方式：

- 检测虫体、蚧类、微小病斑时采用局部高分辨率处理。
- 不要仅为了“创新”无条件增加P2头；先统计bbox面积分布和AP_small。

优点：对微小虫体/病斑直接有效。  
缺点：若器官占画面较大，会增加成本而收益有限。  
优先级：B/C，取决于标注尺度统计。

关键论文：

- QueryDet：https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html
- CFINet：https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_Small_Object_Detection_via_Coarse-to-fine_Proposal_Generation_and_Imitation_Learning_ICCV_2023_paper.html
- AutoFocus：https://openaccess.thecvf.com/content_ICCV_2019/html/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.html

### 6. 检测—分割统一方法

代表方法：Mask R-CNN、Mask2Former、Mask DINO、OneFormer、SAM/EfficientSAM。

- Mask R-CNN适合“器官框+可选掩膜”的渐进标注方式。
- Mask2Former统一语义、实例和全景分割，但仍需任务级训练和掩膜标签。
- Mask DINO同时支持检测和多种分割，可在研究后期将器官框扩展为病斑/器官掩膜。
- OneFormer支持一次训练处理多种分割任务，但工程和显存成本更高。
- SAM不提供病害类别语义，适合预标器官/病斑形状，不适合单独负责病名识别。
- EfficientSAM比原SAM轻，适合作为App或标注端候选。

迁移方式：

- 全量只标bbox。
- 选100—200张高质量子集，用SAM预掩膜+人工修正。
- 对比box-only、box+mask联合监督是否提高分类和背景鲁棒性。

优点：可解释、能计算严重度、可逐步扩展。  
缺点：标注成本高，黄化和煤污等病斑边界主观。  
优先级：B；需要严重度时升为A。

关键论文：

- Mask R-CNN：https://openaccess.thecvf.com/content_ICCV_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html
- Mask2Former：https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf
- Mask DINO：https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Mask_DINO_Towards_a_Unified_Transformer-Based_Framework_for_Object_Detection_CVPR_2023_paper.pdf
- OneFormer：https://openaccess.thecvf.com/content/CVPR2023/html/Jain_OneFormer_One_Transformer_To_Rule_Universal_Image_Segmentation_CVPR_2023_paper.html
- SAM：https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html
- EfficientSAM：https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_EfficientSAM_Leveraged_Masked_Image_Pretraining_for_Efficient_Segment_Anything_CVPR_2024_paper.html

### 7. 视觉基础模型与细粒度识别

代表方法：DINOv2、part matching、原型分类、CLIP/提示学习。

- DINOv2提供通用图像级和patch级表征，适合1000张规模下冻结或部分微调。
- CVPR 2025细粒度few-shot研究显示，局部part特征明显优于只用全局图像特征，DINOv2对分割质量变化相对稳。
- 2026田间植物病害对比研究显示DINOv2+原型/混合头具有较强表现，但堆叠多个大骨干不适合单卡部署。
- CLIP擅长语义，但细粒度病斑纹理未必优于DINOv2；除非App确实提供结构化症状词，否则不应把类别描述复制成文本输入制造标签泄漏。

迁移方式：

- 将检测框裁出的叶/果送入DINOv2-S/ViT-S做二阶段细分类。
- 用DINOv2 patch聚类选择病斑候选区域。
- 采用全局器官特征+局部part特征，而不是只看整框平均特征。

优点：小数据、跨域和局部纹理潜力高。  
缺点：二阶段系统推理更慢；需要验证是否优于单检测头。  
优先级：A/B，适合作为分类分支而非直接替代检测。

关键论文：

- DINOv2：https://arxiv.org/abs/2304.07193
- Fine-grained Few-Shot Part Matching：https://openaccess.thecvf.com/content/CVPR2025W/FGVC/papers/Black_Fine-grained_Few-Shot_Classification_with_Part_Matching_CVPRW_2025_paper.pdf
- 田间植物病害基础模型对比：https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1860665/full

### 8. 半监督检测

代表方法：Soft Teacher、Semi-DETR。

- Soft Teacher通过EMA教师在线生成伪框，并使用分类分数加权和box jitter筛选可靠回归框。
- Semi-DETR针对DETR一对一匹配与伪标签误差问题，引入阶段式混合匹配、跨视图query一致性和基于匹配成本的伪标签挖掘。

适用条件：公司最终采集远多于1000张，但只能人工标其中200—500张。

迁移方式：

- 先人工标200—300张高质量器官框。
- 其余未标数据进入教师学生训练。
- 专家只复核高置信度冲突样本和测试集。

优点：直接利用App后续未标数据。  
缺点：伪标签会放大病名噪声；若1000张都能标，收益可能不如全监督。  
优先级：B；无标签规模显著扩大后升为A。

关键论文：

- Soft Teacher：https://openaccess.thecvf.com/content/ICCV2021/html/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.html
- Semi-DETR：https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Semi-DETR_Semi-Supervised_Object_Detection_With_Detection_Transformers_CVPR_2023_paper.html

### 9. 跨域检测、域泛化与测试时适应

代表方法：Adaptive Teacher、DG-PLDR、FBR-UDA、TENT、Cascade-DETR。

- Adaptive Teacher结合教师学生、弱强增强和域对抗，适合公开柑橘有标注、公司目标图未完全标注的场景。
- DG-PLDR通过风格不敏感蒸馏和原型匹配减少背景/采集风格依赖，已有植物病害代码。
- FBR-UDA使用叶片前景和田间背景重组，再做无监督域适应，能提供背景干预基线。
- TENT只在测试时更新归一化参数和最小化熵，但连续App流中可能发生错误累积，不应直接上线。

迁移方式：

- 首先做有监督目标微调，这是最稳的基线。
- 再比较multi-source ERM、DG-PLDR、Adaptive Teacher。
- TTA只作为后期跨设备流式适应实验。

优点：直接对应公开柑橘→胡柚/香柚、不同设备/果园。  
缺点：对抗训练和伪标签可能不稳定，必须多种子。  
优先级：A（域泛化/目标微调）；B（Adaptive Teacher）；C（TTA）。

关键论文：

- Adaptive Teacher：https://openaccess.thecvf.com/content/CVPR2022/html/Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection_CVPR_2022_paper.html
- DG-PLDR：https://doi.org/10.1016/j.engappai.2025.111168
- FBR-UDA：https://doi.org/10.1016/j.ecoinf.2025.103579
- TENT：https://openreview.net/forum?id=uXl3bZLkr3c

### 10. 主动学习与低成本标注

代表方法：PPAL、ALWOD、DINOv2聚类/coreset。

- PPAL先按检测难度选择候选，再按实例多样性去冗余；适用于任何主流检测器。
- ALWOD结合模型分歧和不确定性，允许标注者校正候选框而不是全部从零绘制。
- 该类方法特别适合App产生大量健康、相似连拍和重复场景的情况。

迁移方式：

1. 首轮随机+分层标200张。
2. 每轮由模型选100—150张困难且多样的样本。
3. 专家优先处理模型分歧、稀有类别和unknown候选。

优点：降低人工浪费，不依赖具体检测器。  
缺点：需要循环训练和标注平台配合。  
优先级：A，建议作为数据流程固定能力；是否作为论文创新视实验效果决定。

关键论文：

- PPAL：https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Plug_and_Play_Active_Learning_for_Object_Detection_CVPR_2024_paper.html
- ALWOD：https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_ALWOD_Active_Learning_for_Weakly-Supervised_Object_Detection_ICCV_2023_paper.pdf

### 11. 开放世界、未知类和置信度校准

代表方法：OW-DETR、OW-OVD、植物病害开放集、检测置信度校准。

- OW-DETR将未知候选、novelty分类和objectness引入DETR。
- OW-OVD进一步结合开放世界和开放词汇，但工程复杂度明显高于当前第一阶段。
- 检测置信度不能只校准类别，还应考虑定位质量和目标尺度。

迁移方式：

- 第一阶段不必完整复现开放世界框架，只需保留unknown/OOD测试和拒识阈值。
- 第二阶段如果App持续出现新虫害，再做增量类别学习。
- 使用机械伤、缺素、反光、未知病斑作为OOD，不把它们强制塞进病虫害类。

优点：符合真实App不会穷尽所有类别的事实。  
缺点：评价复杂，unknown真值也需要专家。  
优先级：B；作为可靠性评测必须保留，作为主算法暂不优先。

关键论文：

- OW-DETR：https://openaccess.thecvf.com/content/CVPR2022/html/Gupta_OW-DETR_Open-World_Detection_Transformer_CVPR_2022_paper.html
- OW-OVD：https://openaccess.thecvf.com/content/CVPR2025/html/Xi_OW-OVD_Unified_Open_World_and_Open_Vocabulary_Object_Detection_CVPR_2025_paper.html
- 自感知检测器与校准：https://openaccess.thecvf.com/content/CVPR2023/papers/Oksuz_Towards_Building_Self-Aware_Object_Detectors_via_Reliable_Uncertainty_Quantification_and_CVPR_2023_paper.pdf
- 植物病害开放集：https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2021.758027/full

### 12. 长尾、蒸馏和多模态

#### 长尾

- Focal Loss处理前景—背景不平衡。
- Class-Balanced Loss按有效样本数重加权。
- Seesaw Loss抑制头部类对尾部类的负梯度压制，适合真实病虫害长尾。
- 不应在类别尚未统计前先决定损失函数。

关键论文：

- Class-Balanced Loss：https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html
- Seesaw Loss：https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Seesaw_Loss_for_Long-Tailed_Instance_Segmentation_CVPR_2021_paper.html

#### 蒸馏

CrossKD让学生检测头的中间特征进入教师检测头，减少教师监督和人工标签之间的矛盾，可在服务器教师确定后蒸馏到手机模型。

关键论文：https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html

#### 多模态

柑橘图像—文本研究表明结构化症状词可帮助复杂背景和小样本分类。App可采集品种、器官、叶正反面、时期、果园和专家症状选项，但必须是真实采集信息，不能在测试时根据答案生成文本。

关键论文：https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1280365/full

优先级：

- 长尾损失：A/B，按类别分布决定。
- 蒸馏：B，部署阶段。
- 图像+真实结构化字段：B，高潜力但依赖App字段质量。
- 自动根据类别生成描述再做多模态：排除，存在标签泄漏风险。

## 方法优先级总表

| 方法家族 | 当前角色 | 标签要求 | 4060 Ti | 与1000张匹配 | 创新潜力 | 建议 |
|---|---|---|---:|---:|---:|---|
| Faster/Cascade R-CNN | 精度基线/主骨架候选 | bbox | 可行 | 高 | 中 | 必做 |
| YOLOv8/11 | 实时基线/部署 | bbox | 很容易 | 高 | 低-中 | 必做但不预设为主创新 |
| FCOS/RetinaNet | 非YOLO一阶段对照 | bbox | 可行 | 高 | 中 | 二选一 |
| RT-DETR | 实时Transformer基线 | bbox | 可行 | 高 | 中-高 | 必做 |
| DINO/Deformable DETR | 高性能Transformer | bbox | 可行但偏重 | 中 | 高 | RT-DETR不足时做 |
| 分类—定位解耦 | 主创新候选 | bbox | 可行 | 高 | 高 | 强烈推荐验证 |
| DINOv2局部分类 | 二阶段细分类 | bbox/类别 | 可行 | 高 | 中-高 | 推荐 |
| SAM前景辅助 | 数据/前景工具 | bbox或点 | 可行 | 高 | 中 | 不单独作病名模型 |
| QueryDet/CFINet | 小病斑/虫体 | 小框 | 偏重 | 条件性 | 高 | 先统计尺度再决定 |
| Soft Teacher/Semi-DETR | 利用大量未标图 | 部分bbox | 可行 | 取决于未标规模 | 中-高 | 数据>1000且标注不足时 |
| Adaptive Teacher/DG | 跨果园/设备/品种 | 源标注+目标数据 | 可行 | 高 | 高 | 第二阶段重点 |
| PPAL/ALWOD | 降标注成本 | 循环标注 | 可行 | 高 | 中 | 建议固化到流程 |
| Mask R-CNN/Mask DINO | 分割与严重度 | mask子集 | 偏重 | 中 | 高 | 有严重度需求时 |
| OW-DETR/开放集 | unknown和持续学习 | unknown评测 | 偏重 | 中 | 高 | 先做拒识评测后升级 |
| CrossKD | 手机部署 | 教师+学生 | 可行 | 高 | 中 | 最后做 |
| 图像+真实元数据 | 多模态辅助 | App字段 | 可行 | 高 | 高 | 先保证字段真实可靠 |

## 六个可独立成题的方向

### 方向A：病征分类—器官定位解耦检测

核心问题：定位叶/果依赖边缘，分类病虫害依赖内部局部病征，两任务共享同一特征会冲突。

方法候选：Faster R-CNN/RT-DETR + 定位边界分支 + DINOv2局部病征分类分支。  
标签：器官bbox+类别。  
优点：标注简单、问题清楚、与现有数据匹配。  
风险：必须证明分支确实关注不同区域。  
当前推荐：第一优先。

### 方向B：跨品种—跨果园多源域检测

核心问题：公开橙橘与胡柚/香柚、不同设备和果园存在域差。

方法候选：有监督目标微调 + DG-PLDR式风格蒸馏/原型匹配，或Adaptive Teacher。  
标签：公开源域标注+本地bbox。  
优点：能充分利用公开柑橘，论文故事完整。  
风险：公开源域和本地类别交集可能不足。  
当前推荐：与A组合或独立第二优先。

### 方向C：近景教师—稍远学生双视图检测

核心问题：近景病斑清晰但缺场景，稍远图有真实背景但病征细节弱。

方法候选：近景DINOv2分类教师 + 稍远RT-DETR/Faster R-CNN学生 + 类别/局部part一致性。  
标签：20%—30%配对对象，稍远图一次点击或bbox。  
优点：与App采集结合，原创空间大。  
风险：依赖公司实现配对流程和sample_id。  
当前推荐：高潜力验证方向。

### 方向D：主动学习驱动的数据闭环

核心问题：1000张中会有大量健康、连拍和相似样本，平均标注浪费资源。

方法候选：PPAL/ALWOD + 农民初标 + 专家复核 + 迭代训练。  
标签：分轮标注。  
优点：直接服务公司数据采集。  
风险：单独作为算法论文创新可能不足，应与A/B结合。  
当前推荐：固定工程能力。

### 方向E：检测—分割—严重度渐进多任务

核心问题：检测框不能量化受害面积。

方法候选：Mask R-CNN/Mask DINO；全量bbox+少量mask。  
标签：100—200张人工修正掩膜。  
优点：应用价值高。  
风险：部分病虫害没有稳定病斑边界。  
当前推荐：公司确认严重度需求后启动。

### 方向F：开放世界持续病虫害识别

核心问题：App上线后一定遇到未收录类别，闭集模型会误诊。

方法候选：OW-DETR、能量/原型拒识、增量类别更新。  
标签：需要unknown测试和后续新增类。  
优点：问题真实、研究新。  
风险：1000张和当前专家标签可能不足，评价复杂。  
当前推荐：先做unknown拒识接口，数据规模扩大后升主线。

## 推荐的基线筛选矩阵

### 第一轮：200—300张

只跑六个模型，统一配置和预训练权重：

1. YOLOv8n/YOLO11n。
2. Faster R-CNN R50-FPN。
3. Cascade R-CNN R50-FPN。
4. FCOS R50-FPN或RetinaNet R50-FPN。
5. RT-DETR-R18。
6. DINOv2-S裁框分类基线，用于判断二阶段细分类是否必要。

停止条件：

- 如果整图/裁框分类与检测头分类差异<1—2个百分点，不增加二阶段分类器。
- 如果AP_small并不低，不做QueryDet/P2类小目标改造。
- 如果跨拍摄者掉点很小，不把域适应作为主创新。
- 如果bbox标注争议大，退回App点击+近景分类或改为多标签。

### 第二轮：只对前三名扩展

- 分类—定位解耦。
- 前景—背景一致性。
- 公开柑橘预训练/目标微调。
- Focal/Class-Balanced/Seesaw三种长尾处理之一。
- PPAL主动选样。

### 第三轮：由真实需求触发

- 未标数据很多：Soft Teacher/Semi-DETR。
- 需要严重度：Mask R-CNN/Mask DINO。
- 跨设备持续漂移：Adaptive Teacher/TTA。
- 新类别频繁出现：OW-DETR/开放世界。
- 必须手机离线：CrossKD、量化和轻量骨干。

## 修改文件

- `reports/2026-09-05/method-landscape-expanded.md`
- `research/inbox/method-landscape-evidence.csv`
- 协调者后续同步 `coordination/PROJECT_STATE.md`。

## 验证命令与结果

计划执行：

```powershell
python scripts/validate_repository.py
rg -n "优先级|标签|迁移|风险|代码|关键论文" reports/2026-09-05/method-landscape-expanded.md
```

验证结果：仓库校验通过；结构化证据表共42条、覆盖24个方法家族；报告关键词检查覆盖优先级、标签、迁移、风险、代码和关键论文。

## 未完成或阻塞

- 尚未获得公司目标果园的真实病虫害频率，无法确定最终类别。
- App能否保存双视图、点击坐标和结构化字段仍待确认。
- 1000张中可获得多少专家确认bbox与病名尚未量化。
- 方法全景已经建立，但最终主线必须由先导数据的尺度、域差和错误类型决定。

## 对下游的影响

- 方法Agent不能只提交YOLO改进方案，必须覆盖至少两阶段、RT-DETR和分类—定位解耦。
- 数据Agent应统计bbox面积、每图目标数、连拍组、类别长尾和跨设备信息，以决定小目标、长尾和域适应是否必要。
- App需求应保留未来主动学习、双视图和unknown闭环所需的元数据。
- 分割和开放世界保持可扩展接口，但不在第一阶段强制投入全部成本。

## 下一步建议

1. 先由数据Agent给出目标类别交集和标注成本表。
2. 公司提供50—100张新App试拍，验证medium/close定义和点击交互。
3. 专家确认3—4个真实高频类别后，构建200—300张器官框先导集。
4. 完成六模型筛选矩阵，再选A、B或C作为论文主线。
5. 对最终两名候选做三随机种子、跨拍摄者/果园测试和消融。
