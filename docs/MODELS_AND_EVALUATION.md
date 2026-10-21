# 模型适配、训练预算与评价标准

## 一、27个检测配置

三模型：YOLOv8s、RetinaNet-R50-FPN、RT-DETR-R18VD。

每模型九组：
- NCCT ×1（独立平扫基线，同患者NCCT测试）。
- AP ×{REAL,SYN,REAL-AUG,REAL+SYN} ×4。
- PVP ×{REAL,SYN,REAL-AUG,REAL+SYN} ×4。

3×9=27。检测不训练AP+PVP融合模型（分类/预后才做双期融合）。

**AP和PVP的REAL/SYN/REAL-AUG/REAL+SYN组使用对应的独立真实CECT测试集**。

## 二、为什么使用一套COCO标准

源伪标签按NIfTI mask转换为二维XYXY；随后一次性同时导出：
- COCO：RetinaNet、RT-DETR及统一检测指标，类别ID为1；
- YOLO TXT：YOLOv8s，类别索引为0；
- images/{train,val,test}：所有模型共用的无损PNG。

统一图像是8-bit L格式（模型读取后转为3通道）、固定HU窗口。图像没有拼成三期输入；每次只使用一种期相。

## 三、模型实现路径

### YOLOv8s

Ultralytics原生YOLOv8s权重或结构；训练时关闭默认mosaic、mixup、HSV、随机几何变化。REAL-AUG通过预生成单个固定种子的旋转/缩放/平移视图与原图1:1组成训练数据，不依赖YOLO内部augment，以避免数据组之间增强条件不一致。

预测时统一阈值0.001导出含score的框，以便COCO AP和FROC使用不同阈值计算。

### RetinaNet-R50

torchvision.models.detection.retinanet_resnet50_fpn，num_classes=2（索引0=背景，索引1=lesion）。COCO读取到的labels统一1。使用AdamW、AMP、固定target_steps近似控制训练预算；依据Val AP@0.5选择最优.pth。默认可下载ImageNet backbone预训练权重；无外网可--no-pretrained。

### RT-DETR-R18VD

采用 https://github.com/lyuwenyu/RT-DETR 中rtdetr_pytorch官方实现，使用其configs/rtdetr/rtdetr_r18vd_6x_coco.yml。适配器自动继承官方config，覆盖数据位置、output目录和类别配置，然后调用tools/train.py。

- 官方自定义数据设置 remap_mscoco_category: False。
- 本项目COCO category_id=1，因此num_classes=2，索引0闲置；最终预测只保留class1并导出category_id=1。
- 输出checkpoint*.pth；适配器尝试根据日志中的Val AP选取最佳轮次。RT-DETR官方代码会变化，正式使用前需在服务器执行小规模试跑并核对配置加载。
- 预测代码按官方tools/infer.py的加载、deploy和postprocessor逻辑适配；在外部官方repo提供的环境运行。

使用官方R18权重时需确认下载和预训练策略，**不要使用RT-DETR-L代替R18**。

## 四、固定训练预算的解释

使用target_steps≈2000作为首轮可行性试跑，而非医学任务必然最优；每组图片数量N、批量B，自动采用：

    epochs = ceil(target_steps / ceil(N/B))

REAL+SYN及REAL-AUG样本量约两倍，但近似总更新次数相同。各架构保持自己的优化器，横向比较重点是**同一架构内部五组数据来源差异**，不是直接比较不同架构的绝对最优性能。

若使用早停，记录实际训练步数/epoch；REAL+SYN增益不能来自单纯增加训练步数。正式论文建议多随机种子与预注册主指标。

## 五、评价

当前统一评价支持：
- COCO AP@[0.5:0.95]（主指标）。
- COCO AP@0.5及AP@0.75。
- 2D **slice-FROC**：横轴平均每切片假阳性框数；纵轴真实2D病灶框被匹配的比例。每张切片内按IoU>=0.5一对一匹配。

**当前未实现：**
- 真实3D lesion-level FROC（必须有跨层病灶实例ID、病灶尺寸和跨切片聚类真值）。
- 患者级FROC、患者级Bootstrap 95%CI及跨中心校准。
- 小病灶按物理直径分层（需要3D实体或相关人工标注）。
- 3D序列连续性配准。
这些不能仅靠2D伪框可信完成，不在代码中杜撰。

预测JSON样例：
~~~json
[{"image_id": 1, "category_id": 1, "bbox": [150, 180, 25, 32], "score": 0.97}]
~~~

所有模型预测同一数据组、同一split，由本项目metrics.py重新计算相同指标。

## 六、数据泄漏与模型选择

- 划分以patient_id为单位，所有期相和Syn/Real使用同一split。
- 教师训练与最终测试患者严格隔离。
- 验证/测试必须有人工确认标签，若使用--allow-pseudo-eval，只能做代码链路测试。
- 使用验证集确定Checkpoint与置信阈值；测试集只做最终独立评价。
- 生成模型训练患者不可与最终独立测试患者重复（如有重叠则应明确其研究性质和限制）。
- 对比REAL与SYN时数据数量、患者、切片来源、标注流程尽可能一致。不要把未合成成功的样本悄然丢弃。

## 七、试跑优先级

1. YOLOv8s + REAL-AP完整训练/预测/指标，检查BBox和COCO。
2. YOLOv8s + SYN-AP完整训练/预测/指标，重点人工检查Syn对应框。
3. YOLOv8s其余7组。
4. RetinaNet-R50其余9组。
5. RT-DETR-R18VD其余9组。

训练前先运行：python -m pytest -q；再手工叠加数十张BBox可视化，检查转换。
