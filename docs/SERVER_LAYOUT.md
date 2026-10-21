# 服务器实际数据布局：gx_all_registered

## 1. 已确定的源数据路径

患者文件夹根目录：/dataset/dataset/gx_all_registered

~~~text
/dataset/dataset/gx_all_registered/
  A0_1/                         <-- ID就是患者文件夹名称；请核对0/O字符
    ap_registered.nii.gz        <-- REAL-AP
    pvp_registered.nii.gz       <-- REAL-PVP
    ncct_zcrop.nii.gz           <-- 固定相NCCT
    ap_mask_registered.nii.gz   <-- 不读作肿瘤病灶伪标签
    pvp_mask_registered.nii.gz  <-- 不读作肿瘤病灶伪标签
    dp_registered.nii.gz        <-- 当前检测任务不使用
    dp_mask_registered.nii.gz
    ncct_mask_zcrop.nii.gz
  A0_2/
    ...
~~~

外部病灶伪标签目录：
- AP：/dataset/detection/ap
- PVP：/dataset/detection/pvp

**现已确认精确文件名**：每位患者的 AP/PVP 伪标签均为 `<patient_id>.nii.gz`，例如 `/dataset/detection/ap/A0_1004.nii.gz` 与 `/dataset/detection/pvp/A0_1004.nii.gz`。内容必须为肿瘤伪Mask，仍待核验shape/affine与病灶语义。

目前自动扫描器识别两类存储方式，按phase分别匹配：

~~~text
方案A：平铺保存
/dataset/detection/ap/A0_1.nii.gz
/dataset/detection/pvp/A0_1.nii.gz

方案A的其他允许命名（只允许额外的常规token）
/dataset/detection/ap/A0_1_ap_lesion_mask.nii.gz
/dataset/detection/pvp/A0_1_pvp_lesion_mask.nii.gz

方案B：按患者子目录保存
/dataset/detection/ap/A0_1/pseudo.nii.gz
/dataset/detection/pvp/A0_1/pseudo.nii.gz
~~~

注意：上方的 **方案A中 `<patient_id>.nii.gz` 已由服务器清单证实**；其他带后缀和嵌套格式仅为扫描器额外支持的方式。 如果无法唯一匹配，脚本会报告患者与期相，不会擅自关联错误Mask。多个相似ID（A0_1与A0_10）不会发生字符串前缀误匹配。如果注册总队列包含未参加检测任务的其他病例，两期伪标签都不存在时，该患者自动跳过；只有某一期有伪标签时则报错，要求人工核对。

## 2. 自动扫描命令

~~~bash
# 克隆/更新main之后
cd detection
pip install -e '.[all]'

mkdir -p /dataset/downstream_detection/manifests
liverdet scan \
  --images /dataset/dataset/gx_all_registered \
  --labels-ap /dataset/detection/ap \
  --labels-pvp /dataset/detection/pvp \
  --out /dataset/downstream_detection/manifests/cases.csv

head -6 /dataset/downstream_detection/manifests/cases.csv
~~~

**不要传额外的NCCT目录**：它固定从每名患者的 ncct_zcrop.nii.gz 取得。

每名患者输出两行（AP、PVP），CSV字段至少为：
patient_id,phase,image_path,mask_path,ncct_path,label_source

举例（仅示意）：
~~~csv
patient_id,phase,image_path,mask_path,ncct_path,label_source
A0_1,ap,/dataset/dataset/gx_all_registered/A0_1/ap_registered.nii.gz,/dataset/detection/ap/A0_1.nii.gz,/dataset/dataset/gx_all_registered/A0_1/ncct_zcrop.nii.gz,pseudo
A0_1,pvp,/dataset/dataset/gx_all_registered/A0_1/pvp_registered.nii.gz,/dataset/detection/pvp/A0_1.nii.gz,/dataset/dataset/gx_all_registered/A0_1/ncct_zcrop.nii.gz,pseudo
~~~

人工核对病例ID、phase、mask_path；不可用患者目录中的 ap_mask_registered/pvp_mask_registered 替代外部伪标签。

## 3. 先验证NIfTI格式与配准关系

~~~bash
liverdet audit \
  --cases /dataset/downstream_detection/manifests/cases.csv \
  --out /dataset/downstream_detection/manifests/nifti_audit.csv

# 额外读取所有体素（更慢）检查伪标签有多少阳性切片：
liverdet audit \
  --cases /dataset/downstream_detection/manifests/cases.csv \
  --out /dataset/downstream_detection/manifests/nifti_audit_full.csv \
  --inspect-masks
~~~

审计报告包含每例AP/PVP与Mask的shape、affine最大绝对偏差、NCCT形状、可能的空间错配、伪标签非零体素数和阳性层数。遇到error会停止后续工作并保存CSV；NCCT不同网格通常仅警告（后续继续匹配/人工审核）；伪标签零体素标为warning，不等于可靠阴性。

## 4. 下一步：转换2D训练样本

在手动审查并固定患者划分CSV后：

~~~bash
liverdet prepare \
  --cases /dataset/downstream_detection/manifests/cases.csv \
  --splits /dataset/downstream_detection/manifests/splits.csv \
  --workdir /dataset/downstream_detection
~~~

将验证/测试用的训练伪标签替换为**人工确认病灶真值**后，才能用于正式论文评价。
如外部伪标签Mask与图像NIfTI具有不同shape或affine，prepare会立刻报错；必须先采用正确的空间变换进行重采样，不能忽视几何差异。

## 5. 核对新增病例伪标签文件名

~~~bash
find /dataset/detection/ap -maxdepth 3 -type f -name '*.nii.gz' | head -20
find /dataset/detection/pvp -maxdepth 3 -type f -name '*.nii.gz' | head -20
~~~

现有清单为患者ID直接命名，扫描器已直接支持。若以后引入不同命名的数据，只需调整inventory.py匹配规则，不影响模型代码。

## 6. 数据质量与后续五组实验

- NCCT：同一病例ncct_zcrop.nii.gz；需审核病灶位置映射。
- REAL：对应ap_registered/pvp_registered。
- SYN：从同患者ncct_zcrop匹配位置定向合成；人工检查合成/框位置。
- REAL-AUG：REAL加固定随机旋转缩放等增强视图。
- REAL+SYN：REAL和对应SYN 1:1。

AP和PVP分别训练，三模型共27配置。患者目录中的NCCT固定相与配准后AP/PVP可能仍存在局部错配；请对匹配结果进行人工QC。本次更新只适配**路径发现**，不改变合成器、训练模型或配准算法。
