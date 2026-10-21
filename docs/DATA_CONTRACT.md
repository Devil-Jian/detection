# 数据格式、空间坐标、伪标签及质控规范

## A. 源数据清单

| 数据 | 当前已知真实路径 | 推荐格式 | 状态 |
|---|---|---|---|
| Real-AP/PVP | /dataset/dataset/gx_all_registered | 3D NIfTI .nii.gz | 用户提供 |
| AP伪标签 | /dataset/detection/ap | NIfTI病灶Mask | 已确认NIfTI扩展名，具体文件名和Mask内容待核验 |
| PVP伪标签 | /dataset/detection/pvp | NIfTI病灶Mask | 已确认NIfTI扩展名，具体文件名和Mask内容待核验 |
| NCCT | /dataset/dataset/gx_all_registered/<patient_id>/ncct_zcrop.nii.gz | 3D NIfTI | 已确认路径 |
| Syn-AP/PVP | 外部生成器产出 | 8-bit灰度PNG（与NCCT切片同网格） | 本仓库不包含生成模型 |

**原始数据永不覆盖。** 处理结果默认保存在 /dataset/downstream_detection。
CT像素强度需要先确认NIfTI的数值是不是HU。当前预处理假定HU，统一窗口[-100,300] HU并映射uint8。

患者目录已确定为 `gx_all_registered/<patient_id>/`；真实AP与PVP分别为 `ap_registered.nii.gz`、`pvp_registered.nii.gz`，NCCT为 `ncct_zcrop.nii.gz`。患者文件夹内的 `ap_mask_registered.nii.gz` 等不是外部病灶伪标签目录。扫描会从 `/dataset/detection/ap`、`/dataset/detection/pvp` 严格匹配患者的NIfTI标签（平铺文件名或患者子目录）。详见 [实际服务器目录说明](SERVER_LAYOUT.md)。

## B. cases.csv 输入契约

列：
- patient_id: 字符串ID，不能含歧义。
- phase: ap / pvp。
- image_path: 真实增强体积NIfTI绝对路径。
- mask_path: 对应配准后NIfTI病灶二值Mask；verified_negative病例可为空。
- ncct_path: 同一患者NCCT体积NIfTI绝对路径；Real-only实验可暂时留空。
- split: train / val / test；优先使用独立splits.csv。
- label_source: manual / pseudo / verified_negative。

一位患者可有两行（AP与PVP），但必须属于同一split。一个患者一期多个序列需人工选择唯一输入，不允许自动猜测。

### 伪标签类型转换规则

当前代码自动支持 **NIfTI二值/整数Mask**（所有大于0视为病灶）。每层采用8邻域连通域提取2D BBox。若同一Mask把多个邻近病灶连在一起，可能合并为一个框；需要人工复核。默认过滤像素面积少于4的连通域。

如果存储为：
- COCO JSON：需要按image_id、patient_id、phase、slice_id重新建立对应，不能直接作为NIfTI Mask加载。
- YOLO TXT：需要知道切片索引和真实图像尺寸，并将归一化中心xywh转到像素XYXY。
- 像素坐标JSON：必须弄清是原始坐标还是配准后坐标。
- 3D病灶实例Mask：如果多个病灶需保持不同ID和跨层链接，当前2D连通域方案尚未保留跨层实例ID。

此版本对上述三种非NIfTI格式会**显式报错**，避免把格式不匹配解释为无病灶。

## C. 坐标变换和影像预处理

- 图像体素索引顺序 NIfTI: [x, y, k]。
- 2D PNG像素坐标: [row=y, col=x]，读取第k层后执行 transpose。
- Real标签BBox存于 **Real-CECT二维像素空间**，格式 xyxy: [x_min, y_min, x_max, y_max]。
- COCO导出框：左上角加宽高 [x_min, y_min, width, height]；类别1。
- YOLO导出：归一化 [class=0,cx,cy,w,h]。
- 图像固定HU窗口裁剪后保存8-bit L格式PNG；训练器转换RGB张量。不同数据组不单独重估窗口。
- Affine及shape不匹配的NIfTI Mask不能直接索引。正确做法是先用最近邻重采样Mask，再做BBox提取；自动处理器当前严格拒绝不一致的输入。

## D. 阴性切片与伪标签质量

**教师没有检出的切片不等于真实阴性。**

允许的阴性来源：
1. 完整人工Mask确认无病灶的层面。
2. trusted_negatives.csv人工审核清单（patient_id,phase,slice_index）。
3. confirmed negative病例在cases.csv标记verified_negative。

没有人工确认的伪标签空切片会被跳过，不会写空BBox进入训练。正常患者可用verified_negative标签进入训练，但应核对是否确实无病灶。

可加入teacher confidence过滤与ignore区域等增强功能，目前基础代码把给定的Mask作为训练监督来源，**不自动修正教师漏检与假阳性**。

## E. splits.csv 患者划分

~~~csv
patient_id,split
P001,train
P002,val
P003,test
~~~

患者划分必须先于切片采样；各期相、各种来源的全部数据沿用同一划分。真实CECT的患者不能在训练集而其合成图像进入测试集。不要将同一患者不同切片当独立病例拆分。

人工验证/测试要求：该split每张纳入评价的切片必须有可靠真值，包括未标记病灶区域的正确阴性判定。原始教师伪标签不可直接充当最终评价金标准。若要计算全面临床敏感度，测试集应包含全肝所有合格切片，而不是仅教师阳性层。

## F. slices.csv 中间结果

| 字段 | 意义 |
|---|---|
| sample_id | <patient_id>_<phase>_<slice_index:04d>，唯一ID |
| patient_id, phase, split | 患者、期相及划分 |
| slice_index | 原始/已配准期相NIfTI第三维索引 |
| real_path | 8-bit灰度REAL PNG |
| source_volume | Real体积NIfTI路径，匹配时读取其affine |
| boxes_real | JSON字符串，XYXY病灶框列表 |
| label_source | manual/pseudo/verified_negative |
| ncct_volume | 对应NCCT NIfTI |
| ncct_slice, ncct_path | 匹配后的NCCT索引及PNG |
| boxes_ncct | Affine投影到NCCT平面的XYXY框 |
| match_nmi, match_offset_mm | 初步图像/物理匹配指标 |
| match_qc | unmatched、pending、pass、fail |
| syn_path | 对应外部合成PNG |
| boxes_syn | Syn输出空间XYXY框；当前假设Syn与NCCT网格一致 |

多病灶一张切片存多个BBox，JSON形如：[[11,20,30,40],[80,90,102,123]]。

## G. NCCT匹配的边界

目前代码为 **Affine初始化 + ±15mm NCCT邻域搜索 + 全图NMI与距离惩罚**。它不是先进配准算法；未执行肝脏ROI语义匹配、呼吸运动形变矫正或跨层单调动态规划。

建议在正式使用前：抽查患者、病灶及肝边界；检查Syn病灶是否位于投影BBox内；不可靠样本记录fail或重新配准。manual review CSV的pass并不保证没有错误，但可提供可审计的QC链。

### NCCT基线的特殊情况

单一NCCT基线优先使用AP对应的标签集合，缺失AP的患者可使用PVP；若不同CECT目标层面匹配到同一NCCT切片造成标签冲突，导出会中止，要求先审查并合并标签。不能仅为凑齐NCCT组而随意覆盖。

## H. 外部生成器接口

synthesis_requests.csv：
~~~csv
sample_id,patient_id,phase,ncct_input,syn_output,qc
P001_ap_0042,P001,ap,/dataset/downstream_detection/data/ncct/P001/0044.png,/dataset/downstream_detection/synthetic_outputs/ap/P001/0042.png,pass
~~~

你的生成模型应把ncct_input转换成对应期相的8-bit单通道PNG，**尺寸与源NCCT PNG相同**。若生成器输出[-1,1]浮点tensor或HU NIfTI，需要先按固定的、经过核实的逆归一化映射导出，而不能直接乘255。

同一患者同一NCCT输入可以分别生成Syn-AP与Syn-PVP，但检测主线两期独立模型，不需要双期成对输入检测网络。

## I. 数据源可追溯与不允许的行为

- 不允许把不同affine的Real框直接拷贝到Syn图像。
- 不允许在测试集上用教师重新补伪标签作为评价真值。
- 不允许将生成器见过的患者作为独立下游泛化测试样本却不说明。
- 不允许因为某些Syn未生成而悄悄删除REAL训练患者；组间对比应同人同位置。
- 不允许将2D slice-FROC自动称为3D lesion-FROC。
