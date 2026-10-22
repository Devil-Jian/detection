# 肝脏病灶目标检测｜NCCT / Real-CECT / Syn-CECT 下游实验

**目标：** 在相同患者划分和统一检测标注下，比较非增强CT（NCCT）、真实增强CT（REAL）、合成增强CT（SYN）、传统增强（REAL-AUG）与真实+合成1:1（REAL+SYN）的肝脏病灶检测性能。

> **实验定位**：约300例，2D单类别 liver_lesion，AP与PVP分别训练。增强相关组均在真实相应期相测试集评价；NCCT在相同测试患者的NCCT图像上评价。本仓库**不是**合成模型本体；使用已有NCCT→CECT模型输出合成图像后再接入。**未运行真实服务器数据训练**，须先核实你现有伪标签格式与几何空间。

## 当前主线：仅过滤训练伪标签中的2D微小连通域（不使用Liver Mask）

**默认不对伪标签执行肝脏ROI交集。** Train伪标签直接在对应AP/PVP NIfTI每张切片提取8连通域，删除小于 `--min-component-pixels` 指定的前景面积（默认4像素）的连通域，然后生成BBox。阈值按前景像素数量，而不是BBox面积；建议在真实数据上抽查被删除的候选。

**Val/Test人工肿瘤真值始终不受该阈值影响**，包括1像素组件。伪标签去噪后没有病灶框的切片不能自动当作阴性。原始影像及Mask NIfTI不会被修改。

`liverdet prepare` 默认不要求 `ap_mask_registered.nii.gz` 或 `pvp_mask_registered.nii.gz`，写入 `small_component_filter_qc.csv` 记录过滤的组件数。只有明确传入 `--filter-pseudo-liver-mask` 才启用历史肝Mask裁剪方案；`liverdet audit` 同样默认不检查肝Mask，历史检查需 `--check-liver-mask`。

**[当前执行命令和数据质控](docs/SMALL_COMPONENT_FILTER.md)**。旧 Liver ROI 过滤作为可选对照保留于 [LIVER_PSEUDO_FILTER.md](docs/LIVER_PSEUDO_FILTER.md)。


---

## 目标检测专用Case ID表：伪标签Train、人工标签Val/Test（优先使用）

**优先使用新命令 liverdet build-cohorts，而不是scan扫描全部注册病例。** 你的病例清单含 452 名患者：344名pseudo_label（其余342名mask路径待补全）+108名ground_truth。此统计来自用户提供的列表，实际可用Train人数取决于服务器伪Mask是否齐全。

~~~bash
BASE=/dataset/downstream_detection
mkdir -p "$BASE/manifests"

# 请将私人Case ID CSV上传到服务器指定私有路径；不要上传完整名单到公开代码仓库。
liverdet build-cohorts \
  --case-list /dataset/detection/detection_case_ids.csv \
  --workdir "$BASE" \
  --labels-ap /dataset/detection/ap \
  --labels-pvp /dataset/detection/pvp \
  --val-fraction 0.5 --seed 42

liverdet audit --cases "$BASE/manifests/cases.csv" \
  --out "$BASE/manifests/nifti_audit.csv" --inspect-masks

liverdet prepare --cases "$BASE/manifests/cases.csv" \
  --splits "$BASE/manifests/splits.csv" --workdir /dataset/downstream_detection_cc4 \
  --min-component-pixels 4
~~~

- TRAIN：仅纳入两期伪Mask完整的pseudo_label病例；仅训练用，不可进Val/Test。
- VAL/TEST：108名ground_truth患者按患者划分54/54，读取患者目录中的ap_tumor_mask_registered.nii.gz和pvp_tumor_mask_registered.nii.gz。人工GT不会被伪标签替代。
- 对pending，按患者ID尝试查找 /dataset/detection/ap/<patient_id>.nii.gz 和PVP对应文件。默认缺失报错；确认原因后才可加 --skip-missing-pseudo 排除缺失的Train病例。
- 输出 cases.csv、splits.csv、cohort_inventory.csv。**完全不把地面真实标签用于student检测器训练**。
- 如GT病例曾用于教师模型训练，必须重新核查教师-Student测试集独立性。

**[查看详细病例划分及数据说明](docs/CASE_LIST_SPLIT.md)**。

---

## 真实服务器目录已适配

已支持从患者根目录 `/dataset/dataset/gx_all_registered/<patient_id>/` 自动读取 `ap_registered.nii.gz`、`pvp_registered.nii.gz`、`ncct_zcrop.nii.gz`，并分别从 `/dataset/detection/ap`、`/dataset/detection/pvp` 匹配NIfTI病灶伪标签。患者文件夹内的 `ap_mask_registered.nii.gz` / `pvp_mask_registered.nii.gz` **不会**被当成外部病灶标签。

无需再提供 `--ncct-root`。具体示例、检查命令和伪标签命名规则见 **[实际服务器路径说明](docs/SERVER_LAYOUT.md)**。已确认外部AP/PVP伪标签按 `<patient_id>.nii.gz` 平铺保存，例如 `A0_1004.nii.gz`。伪标签与影像的shape/affine及病灶内容仍需通过 `liverdet audit` 核验。若配准总队列大于检测队列，扫描器仅纳入AP和PVP两期均具有唯一外部伪标签的患者；两期均无伪标签的患者自动跳过，一期有标签另一期缺失则报错。


### 扫描后先运行数据完整性检查

已确认实际标签命名为：`/dataset/detection/ap/A0_1004.nii.gz`、`/dataset/detection/pvp/A0_1004.nii.gz`（其他患者同理）。**扫描后建议先运行audit，不要立刻大量提取2D数据。**

~~~bash
BASE=/dataset/downstream_detection
liverdet scan \
  --images /dataset/dataset/gx_all_registered \
  --labels-ap /dataset/detection/ap \
  --labels-pvp /dataset/detection/pvp \
  --out "$BASE/manifests/cases.csv"

# 检查文件存在、影像与伪标签shape/affine对应关系（仅读元数据，速度较快）
liverdet audit --cases "$BASE/manifests/cases.csv" \
  --out "$BASE/manifests/nifti_audit.csv"

# 可选：读取伪Mask全体素，统计阳性切片和非零体素（时间更长）
liverdet audit --cases "$BASE/manifests/cases.csv" \
  --out "$BASE/manifests/nifti_audit_full.csv" --inspect-masks
~~~

当报告中含`status=error`时命令会报错，但检查CSV仍会保存以便定位失败患者。NCCT与配准后CECT不完全同网格只会提示warning，需在后续NCCT匹配阶段人工质控；**AP/PVP伪标签Mask与相应Real-CECT的shape/affine不一致属于error，必须先修复。** 伪标签中0病灶的层面不自动作为阴性。

完整命名和检查说明：[docs/SERVER_LAYOUT.md](docs/SERVER_LAYOUT.md)。

## 0. 实施状态及安全边界

| 模块 | 状态 | 说明 |
|---|---|---|
| NIfTI图像+NIfTI伪Mask→2D PNG/BBox | 已实现 | 默认伪标签仅过滤面积过小的2D连通域；GT不做面积过滤；Liver ROI裁剪为可选 |
| 病例扫描和患者级拆分 | 已实现 | 自动扫描只对常见命名有效；不符请手工准备cases.csv |
| 可靠阳性/阴性采样 | 已实现 | 未检出的伪标签空层不当作阴性 |
| NCCT受限相似性匹配 | 已实现初版 | 物理位置+局部全图NMI；结果**必须人工审核**；尚未实现肝脏ROI/序列动态规划 |
| 生成请求清单与合成结果接入 | 已实现 | **不包含你的合成模型权重与实际推理调用** |
| COCO+YOLO五组导出 | 已实现 | REAL-AUG固定种子生成1:1几何增强视图 |
| YOLOv8s训练/预测 | 已实现 | 首次运行可能下载预训练权重 |
| RetinaNet-R50训练/预测 | 已实现 | Torchvision |
| RT-DETR-R18VD训练/预测 | 已实现官方代码适配层 | 需要另行安装lyuwenyu/RT-DETR PyTorch项目；不能以RT-DETR-L替代 |
| COCO mAP与2D slice-FROC | 已实现 | 真正3D病灶级FROC需额外的跨层病灶ID人工真值 |
| 合成病例上的完整真实数据效果 | 尚未验证 | 本仓库没有挂载你的服务器数据/GPU |

## 1. 仓库目录

~~~text
configs/experiment.yaml             # 示例路径与实验设置
src/liverdet/
  inventory.py                      # 扫描AP/PVP与伪标签并建立cases.csv
  prepare.py                        # 3D NIfTI -> 2D PNG + XYXY病灶框
  matching.py                       # NCCT局部匹配、QC、合成请求、接入Syn PNG
  export.py                         # 五组导出：COCO + YOLO
  train.py                          # YOLOv8 / RetinaNet训练与预测
  rtdetr.py                         # 官方RT-DETR-R18适配器
  metrics.py                        # 统一COCO AP与2D slice-FROC
  cli.py                            # 命令行入口
tests/test_pipeline.py              # 合成NIfTI小样本端到端测试
docs/DATA_CONTRACT.md               # 文件格式、坐标、标签与质控约定
docs/MODELS_AND_EVALUATION.md        # 三检测器、实验矩阵、指标说明
~~~

## 2. 服务器环境

建议Python 3.10或3.11、Linux GPU。请按本机CUDA版本选择PyTorch，不要盲目覆盖现有CUDA环境。

~~~bash
git clone https://github.com/Devil-Jian/detection.git
cd detection
# 当前代码已合并到main；无需切换旧开发分支。

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
python -m pytest -q
liverdet --help
~~~

默认输入目录（仅引用，**不移动、修改原始数据**）：

~~~text
/dataset/dataset/gx_all_registered   # AP/PVP NIfTI
/dataset/detection/ap                # AP伪标签
/dataset/detection/pvp               # PVP伪标签
<用户提供的NCCT目录>                  # 对应原始NCCT NIfTI
<用户已有生成器>                     # 由NCCT生成Syn-AP/PVP
~~~

### 必须先检查伪标签格式

当前自动读取假定AP/PVP伪标签是和影像几何一致的 **3D NIfTI Mask（.nii/.nii.gz）**。如果你保存的是YOLO TXT、JSON、PNG或别的坐标空间，**不要直接执行prepare**；请先转成Mask NIfTI或改写标签解析器，且确认patient_id与slice_index一致。

~~~bash
find /dataset/dataset/gx_all_registered -maxdepth 3 -type f | head -30
find /dataset/detection/ap -maxdepth 3 -type f | head -20
find /dataset/detection/pvp -maxdepth 3 -type f | head -20
~~~

## 3. 输入→处理→输出总数据流

~~~text
真实AP/PVP（.nii.gz）    对应病灶伪Mask（.nii.gz）
           \                  /
              scan -> cases.csv
                     |
         指定患者级Train/Val/Test
                     |
               prepare
                     |
      Real-AP/Real-PVP PNG + BBox + slices.csv
                     |
     ┌───────────────┴─────────────────┐
     |                                 |
 REAL可直接导出                  NCCT物理坐标初定位
                                       |
                              ±15mm候选 + NMI排序
                                       |
                                 人工QC标记pass
                                       |
                           NCCT PNG + boxes_ncct
                                       |
                              synthesis_requests.csv
                                       |
                    <外部生成器：NCCT->Syn AP/PVP>
                                       |
                        Syn PNG（与NCCT相同XY网格）
                                       |
                                    attach-syn
                                       |
                   导出五组COCO + YOLO Dataset
                                       |
                   训练YOLO / RetinaNet / RT-DETR
                                       |
                         COCO预测JSON + 统一mAP
~~~

详细数据契约参见 [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md)。

## 4. 准备病例清单

### 4.1 自动发现（适用于常规文件命名）

~~~bash
liverdet scan \
  --images /dataset/dataset/gx_all_registered \
  --labels-ap /dataset/detection/ap \
  --labels-pvp /dataset/detection/pvp \
  --out /dataset/downstream_detection/manifests/cases.csv
~~~

自动扫描严格读取每位患者的ap_registered.nii.gz、pvp_registered.nii.gz、ncct_zcrop.nii.gz，并按相同patient_id读取外部病灶伪标签（如A0_1004.nii.gz）。无法唯一关联的文件会报错，不能忽略。

### 4.2 手工病例CSV（更稳妥，推荐）

~~~csv
patient_id,phase,image_path,mask_path,ncct_path,split,label_source
P001,ap,/dataset/dataset/gx_all_registered/P001/ap.nii.gz,/dataset/detection/ap/P001.nii.gz,/dataset/ncct/P001.nii.gz,train,pseudo
P001,pvp,/dataset/dataset/gx_all_registered/P001/pvp.nii.gz,/dataset/detection/pvp/P001.nii.gz,/dataset/ncct/P001.nii.gz,train,pseudo
P002,ap,/dataset/dataset/gx_all_registered/P002/ap.nii.gz,/dataset/detection/ap/P002.nii.gz,/dataset/ncct/P002.nii.gz,val,manual
P002,pvp,/dataset/dataset/gx_all_registered/P002/pvp.nii.gz,/dataset/detection/pvp/P002.nii.gz,/dataset/ncct/P002.nii.gz,val,manual
~~~

上述只是格式示例，**不代表你的服务器上存在这些患者或路径**。其中label_source取值为manual、pseudo或verified_negative。Val/Test需要人工确认的阳性病灶与可靠阴性。建议显式提供患者划分CSV（patient_id,split），不要在正式论文中依赖随机拆分。

### 4.3 切片提取与BBox

~~~bash
liverdet prepare \
  --cases /dataset/downstream_detection/manifests/cases.csv \
  --workdir /dataset/downstream_detection \
  --splits /dataset/downstream_detection/manifests/splits.csv \
  --hu-min -100 --hu-max 300
~~~

运行后：

~~~text
/dataset/downstream_detection/
  manifests/cases_with_splits.csv
  manifests/slices.csv
  data/real/ap/P001_ap_0042.png
  data/real/pvp/P001_pvp_0042.png
~~~

PNG统一8位单通道、固定窗口（默认 -100至300 HU）。检测训练时读取灰度并复制到3通道。Mask逐轴向k切片，得到多个2D连通域BBox。注意：**NIfTI数组[x,y,k]在PNG保存前转置为图像[y,x]**。Mask和图像shape/affine不一致时直接中断。

**阴性切片**：人工Mask中无病灶的层面，或由trusted_negatives.csv（patient_id,phase,slice_index）明确审过的层面。教师未检出**不等于阴性**。当伪Mask只有阳性时，不会擅自补齐阴性，可用：

~~~bash
liverdet prepare \
  --cases ... --workdir ... \
  --trusted-negatives /dataset/downstream_detection/manifests/trusted_negatives.csv
~~~

当前默认保留所有已确认的切片，未自动强制阳性:阴性=1:1。比例应在训练采样阶段按单期和来源控制，并记录患者病灶负荷；不可把验证/测试集下采样成1:1后作为总体检出率。

## 5. 匹配NCCT并接入已有合成模型

由于NCCT为固定相、AP/PVP为移动相，匹配先使用NIfTI affine中的物理空间信息确定NCCT初始层；在±15mm内搜索候选层，按NMI与距离惩罚排序。本版只实现**全图NMI**，尚不包含自动肝脏分割ROI、图像局部形变校正、序列动态规划或完整临床配准评估。

~~~bash
liverdet match-ncct \
  --manifest /dataset/downstream_detection/manifests/slices.csv \
  --workdir /dataset/downstream_detection \
  --radius-mm 15
~~~

每条匹配默认标为pending。需要对图像重叠、肝门、血管及病灶位置进行QC，并用CSV记录：

~~~csv
sample_id,match_qc
P001_ap_0042,pass
P001_pvp_0042,pass
P004_ap_0050,fail
~~~

~~~bash
liverdet review-matches \
  --manifest /dataset/downstream_detection/manifests/slices.csv \
  --reviews /dataset/downstream_detection/manifests/matching_reviews.csv
~~~

在不具备形变校正的条件下，把Real BBox通过Affine转换至NCCT平面是**近似映射**。必须对Syn训练标签进行实际可视化审核；尤其当真实增强切片与NCCT存在局部错层时，不能仅靠NMI最高和像素框变换认定标签准确。

### 5.1 输出合成任务清单

~~~bash
liverdet synthesis-requests \
  --manifest /dataset/downstream_detection/manifests/slices.csv \
  --syn-root /dataset/downstream_detection/synthetic_outputs \
  --out /dataset/downstream_detection/manifests/synthesis_requests.csv
~~~

CSV包含：sample_id、patient_id、phase、ncct_input、syn_output、qc。

**你自己的生成器需要读取ncct_input并生成对应期相图像**。生成输出必须使用与匹配NCCT PNG完全一致的W×H与8-bit灰度处理规则。推荐如下结构：

~~~text
/dataset/downstream_detection/synthetic_outputs/
  ap/P001/0042.png
  pvp/P001/0042.png
~~~

此处文件名中的0042是**目标真实CECT的slice_index**，而不是matched NCCT的index。真正的NCCT输入见请求CSV。

### 5.2 导入合成结果

~~~bash
liverdet attach-syn \
  --manifest /dataset/downstream_detection/manifests/slices.csv \
  --syn-root /dataset/downstream_detection/synthetic_outputs
~~~

会校验合成PNG是否存在、是否uint8灰度、图像尺寸是否与对应NCCT一致、是否通过匹配质控，完成后写回syn_path与boxes_syn。没有生成器及NCCT路径时，可先跑REAL实验。

## 6. 五组数据怎么构建

| 配置名 | 训练输入 | 标签 | 验证/测试 |
|---|---|---|---|
| **NCCT** | 真实NCCT切片（一个独立基线） | 人工审核的NCCT对应框 | 相同测试患者的NCCT |
| **REAL** | 真实AP或真实PVP切片 | 对应真实期相框 | 相同相位Real-CECT |
| **SYN** | 匹配NCCT生成的Syn-AP或Syn-PVP | 经空间转换及QC的框 | 相同相位Real-CECT |
| **REAL-AUG** | 真实切片+确定性随机几何增强视图（1:1） | 变换后的框 | 相同相位Real-CECT |
| **REAL+SYN** | 真实与配对Syn切片（1:1） | 各自坐标空间的框 | 相同相位Real-CECT |

- **1:1是Real:Syn或Real:Aug，绝不是AP:PVP**，每个AP、PVP都单独建模。
- REAL-AUG使用固定随机种子，在导出时保存旋转±10°、缩放0.9–1.1、平移±5%的视图。导出两份样本不代表增加独立患者。
- REAL+SYN的Real和Syn来自同一患者及目标解剖位置，必须在相同split中。
- 验证与测试均用Real影像，独立于训练来源；NCCT除外。
- 导出的COCO是唯一的评价真值，YOLO txt从COCO同时导出；**三模型完全使用同一个数据组目录**。

~~~bash
BASE=/dataset/downstream_detection
MANIFEST=$BASE/manifests/slices.csv

# 可以在SYN尚未完成时先导出REAL：
liverdet export --manifest "$MANIFEST" --out-root "$BASE/exports" \
  --group cect_real --phase ap
liverdet export --manifest "$MANIFEST" --out-root "$BASE/exports" \
  --group cect_real --phase pvp

# 完成NCCT与Syn匹配QC/合成后，按组导出
for phase in ap pvp; do
  for group in cect_syn cect_real_aug cect_real_syn; do
    liverdet export --manifest "$MANIFEST" --out-root "$BASE/exports" \
      --group "$group" --phase "$phase"
  done
done
liverdet export --manifest "$MANIFEST" --out-root "$BASE/exports" --group ncct
~~~

若只有教师伪标签而暂时没有人工Val/Test标注，为验证代码功能可加 **--allow-pseudo-eval**，但导出目录会标注SMOKE_TEST_ONLY，**这些伪标签测试指标不能用于论文结论**。

导出样例：

~~~text
exports/cect_real_ap/
  dataset.yaml
  counts.csv
  images/train/*.png
  images/val/*.png
  images/test/*.png
  labels/train/*.txt             # YOLO归一化xywh，单类别0
  labels/val/*.txt
  labels/test/*.txt
  annotations/train.json        # COCO bbox xywh，category_id=1
  annotations/val.json
  annotations/test.json
~~~

## 7. 模型训练与预测

### 7.1 YOLOv8s：首个必须跑通的基线

~~~bash
BASE=/dataset/downstream_detection
DATA=$BASE/exports/cect_real_ap

liverdet train --model yolov8s --dataset "$DATA" \
  --out "$BASE/runs/yolov8s_ap_real_seed42" \
  --steps 2000 --batch 8 --device 0 --seed 42

liverdet predict --model yolov8s \
  --dataset "$DATA" --split test \
  --checkpoint "$BASE/runs/yolov8s_ap_real_seed42/weights/best.pt" \
  --out "$BASE/results/yolov8s_ap_real_preds.json" --device 0

liverdet evaluate --dataset "$DATA" --split test \
  --predictions "$BASE/results/yolov8s_ap_real_preds.json" \
  --out "$BASE/results/yolov8s_ap_real_metrics.json"
~~~

默认初始权重yolov8s.pt（首次运行可能联网下载），用--no-pretrained从零训练。所有组关闭YOLO内置mosaic/mixup/几何与颜色随机增强，以避免REAL与REAL-AUG的比较混杂。

### 7.2 RetinaNet-R50（Torchvision）

~~~bash
liverdet train --model retinanet_r50 --dataset "$DATA" \
  --out "$BASE/runs/retinanet_ap_real_seed42" \
  --steps 2000 --batch 8 --device 0

liverdet predict --model retinanet_r50 --dataset "$DATA" \
  --checkpoint "$BASE/runs/retinanet_ap_real_seed42/best.pth" \
  --out "$BASE/results/retinanet_ap_real_preds.json" --device 0
~~~

RetinaNet使用torchvision ResNet50-FPN单类别目标、背景类索引0、目标类索引1。初始权重为ImageNet Backbone（可以--no-pretrained避免下载）。所有组固定相同目标优化步数，按每组图片数自动换算epoch上限。

### 7.3 RT-DETR-R18VD（官方PyTorch实现）

本项目**不会把RT-DETR-L伪称R18**。需要单独克隆[官方RT-DETR](https://github.com/lyuwenyu/RT-DETR)，并在官方仓库环境中使用其R18VD配置。项目适配器生成数据专用YAML覆盖路径与类别数，之后调用官方tools/train.py。

~~~bash
git clone https://github.com/lyuwenyu/RT-DETR.git /dataset/external/RT-DETR

liverdet train --model rtdetr_r18 \
  --rtdetr-repo /dataset/external/RT-DETR \
  --dataset "$DATA" \
  --out "$BASE/runs/rtdetr_ap_real_seed42" \
  --steps 2000 --batch 8 --device 0
~~~

训练后运行预测（必须提供生成的配置文件）：

~~~bash
liverdet predict --model rtdetr_r18 \
  --rtdetr-repo /dataset/external/RT-DETR \
  --rtdetr-config "$BASE/runs/rtdetr_ap_real_seed42/rtdetr_r18_experiment.yml" \
  --checkpoint "$BASE/runs/rtdetr_ap_real_seed42/checkpoint.pth" \
  --dataset "$DATA" \
  --out "$BASE/results/rtdetr_ap_real_preds.json" --device 0
~~~

**注意：** 官方RT-DETR还依赖其版本特定requirements；请在兼容的PyTorch/CUDA环境安装。当前适配器按官方2026年10月可获取的代码编写，但未在你的GPU执行，首次运行前须确认官方依赖版本。官方raw category_id=1而remap_mscoco_category=false，因此适配器显式设置num_classes=2，类别索引1用于病灶预测。

参见 [docs/MODELS_AND_EVALUATION.md](docs/MODELS_AND_EVALUATION.md)。

## 8. 27组实验矩阵（固定主线）

~~~text
3种网络 × [1组NCCT + 2期相 × 4种增强CECT训练来源]
= 3 × (1 + 8) = 27

YOLOv8s       : ncct(1) + ap REAL/SYN/REAL-AUG/REAL+SYN(4) + pvp(4)
RetinaNet-R50  : 同上
RT-DETR-R18VD  : 同上
~~~

~~~bash
liverdet matrix
~~~

优先级：先跑YOLOv8s REAL-AP（检查标签正确性）→ SYN-AP（检查配准和合成框）→ YOLO全9组 → 其余18组。

## 9. 评价结果的正确解读

统一预测JSON（COCO格式）：

~~~json
[
  {"image_id": 1, "category_id": 1, "bbox": [120, 200, 30, 35], "score": 0.92}
]
~~~

统一评价输出：AP@[0.5:0.95]、AP@0.5、AP@0.75、**slice-FROC**（每张2D切片假阳性数 versus 2D物体敏感度）。本版**尚未实现3D病灶级FROC及患者级Bootstrap置信区间**，因为它们需要跨层病灶关联的人工金标准与患者级统计；不能将2D slice-FROC标成3D lesion-FROC。

Val/Test必须完整保留实际临床阳性/阴性切片分布。若你只给了训练伪标签，评估脚本可以用于调试，但不能宣称临床检出率。

## 10. 数据完整性和论文公平对照

1. 所有同患者的NCCT/REAL/SYN必须落入同一split，任何模型/生成器不能见过最终独立测试患者。
2. REAL和SYN使用相同患者与目标位置，标签来源和映射可追溯；对齐失败不能复制错误BBox。
3. REAL-AUG与REAL+SYN都是原图:新增训练视图=1:1；各模型控制训练预算。合成数据不等于增加独立患者数。
4. 无人工真值的伪标签空层不能标成可靠阴性，尤其是正常肝脏与小病灶切片。
5. 同时记录模型随机种子、预训练情况、训练步数、类别映射、窗口、合成模型版本与匹配QC。
6. 论文最终评估需补充3D病灶关联与患者级Bootstrap，必要时还应增加跨中心测试。

## 11. 常见错误

- **scan找不到数据：** 请检查每名患者的固定文件名ap_registered、pvp_registered、ncct_zcrop及外部伪标签是否可根据patient_id严格匹配。详见docs/SERVER_LAYOUT.md。
- **image/mask geometry mismatch：** 不要强行按numpy索引对齐，先检查shape/affine，必要时重采样Mask（最近邻）。
- **只发现阳性切片：** 教师伪Mask的空层不是阴性；提供受审核阴性CSV。
- **Val/Test must have manual ground truth：** 准备人工标注后再做正式验证。
- **match_qc pending：** 创建review-matches CSV；高NMI不能自动通过。
- **Syn输出没读到：** 检查synthesis_requests.csv中的syn_output实际路径。
- **REAL+SYN少了合成样本：** 需完成所有目标训练位置的合成和质控，不能静默丢失。
- **GPU/依赖版本不兼容：** 分开建环境安装官方RT-DETR；优先跑完YOLO smoke。
- **论文结果不要引用SMOKE_TEST_ONLY导出目录。**

## 12. 尚需确认的真实数据接口

已确认NCCT与Real-AP/PVP同处gx_all_registered患者目录，伪标签为NIfTI格式。**已确认外部伪标签精确文件名为 `<patient_id>.nii.gz`；仍需核验Mask的数值语义以及是否与配准后的AP/PVP具有一致shape/affine。** 如果命名不同需扩展路径解析；如果空间不同需正确重采样。NCCT→CECT生成器接口及人工验证/测试真值也仍需提供。

详细文档：[数据格式/标签/质控](docs/DATA_CONTRACT.md)｜[训练和评价](docs/MODELS_AND_EVALUATION.md)。


## 13. 批量运行27组（先dry-run，后正式）

完成九个数据导出后：

~~~bash
BASE=/dataset/downstream_detection

# 仅预览YOLOv8s的9组，不训练：
python scripts/run_matrix.py \
  --exports "$BASE/exports" --runs "$BASE/runs" --results "$BASE/results" \
  --models yolov8s --dry-run

# 正式顺序运行YOLOv8s的9组，断点续跑：
python scripts/run_matrix.py \
  --exports "$BASE/exports" --runs "$BASE/runs" --results "$BASE/results" \
  --models yolov8s --steps 2000 --batch 8 --device 0 --skip-completed

# 完整27组（需要先安装官方RT-DETR及相关环境）：
python scripts/run_matrix.py \
  --exports "$BASE/exports" --runs "$BASE/runs" --results "$BASE/results" \
  --rtdetr-repo /dataset/external/RT-DETR \
  --steps 2000 --batch 8 --device 0 --skip-completed
~~~

脚本自动从每次训练写出的best_checkpoint.txt读取最优Checkpoint，完成预测及统一评价，并输出 experiment_manifest.csv。当前为**顺序训练**，适合单GPU/独立任务队列；多GPU可在不同进程用--models筛选，务必保证输出目录不冲突。真实数据未经验证前先用 --dry-run 检查所有配置。
