# 目标检测病例清单：伪标签 Train 与人工真值 Val/Test

## 1. 不再只用scan

原有 liverdet scan 会遍历整个 registered 患者根目录，按伪标签路径筛选患者，但不会读取你的病例清单中 cohort_type=ground_truth 的病例。它因此不能正确划分当前研究的Train/Val/Test。

**新命令 liverdet build-cohorts 只以你的Case ID清单为权威名单。** 不遍历所有影像目录决定患者是否入组。

## 2. 你的病例清单CSV格式

~~~csv
patient_id,cohort_type,ap_image,ap_mask,ap_mask_type,pvp_image,pvp_mask,pvp_mask_type,source_folder
A0_1004,pseudo_label,/dataset/dataset/gx_all_registered/A0_1004/ap_registered.nii.gz,/dataset/detection/ap/A0_1004.nii.gz,pseudo_label,/dataset/dataset/gx_all_registered/A0_1004/pvp_registered.nii.gz,/dataset/detection/pvp/A0_1004.nii.gz,pseudo_label,/dataset/dataset/gx_all_registered/A0_1004
A0_1461,pseudo_label,/dataset/dataset/gx_all_registered/A0_1461/ap_registered.nii.gz,,pending,/dataset/dataset/gx_all_registered/A0_1461/pvp_registered.nii.gz,,pending,/dataset/dataset/gx_all_registered/A0_1461
A1_100,ground_truth,/dataset/dataset/gx_all_registered/A1_100/ap_registered.nii.gz,/dataset/dataset/gx_all_registered/A1_100/ap_tumor_mask_registered.nii.gz,ground_truth,/dataset/dataset/gx_all_registered/A1_100/pvp_registered.nii.gz,/dataset/dataset/gx_all_registered/A1_100/pvp_tumor_mask_registered.nii.gz,ground_truth,/dataset/dataset/gx_all_registered/A1_100
~~~

这些是原表中的三个代表性记录，完整病人清单应只保存于服务器私有路径，不要提交到公开GitHub仓库。

原表共452名患者：
- pseudo_label：344名，其中342名Mask路径为空且mask_type=pending，另2名已填写AP/PVP伪标签路径。
- ground_truth：108名，其AP/PVP人工标注文件路径已填写。
- 452个patient_id互不重复。

**pending只表示CSV里尚未填写路径，不代表服务器的伪标签文件不存在。** 程序按精确ID查找 /dataset/detection/ap/<patient_id>.nii.gz 和 /dataset/detection/pvp/<patient_id>.nii.gz。两期均存在才可入Train。缺失时默认报错并写入cohort_inventory.csv；只在显式传入 --skip-missing-pseudo 时排除不完整的Train患者。

人工GT不会使用伪标签替代，读取病例清单中的：
~~~text
/dataset/dataset/gx_all_registered/<patient_id>/ap_tumor_mask_registered.nii.gz
/dataset/dataset/gx_all_registered/<patient_id>/pvp_tumor_mask_registered.nii.gz
~~~

## 3. 默认Train/Val/Test划分

| 划分 | 数据来源 | 人数 |
|---|---|---:|
| Train | pseudo_label | 最多344（以两期实际文件完整的患者数为准） |
| Val | ground_truth | 54 |
| Test | ground_truth | 54 |

默认 --val-fraction 0.5 --seed 42。108名GT患者按固定随机种子划成54/54。AP、PVP共享同一个患者划分，所有切片及NCCT/SYN继承这份划分。

划分一旦确定就冻结，不应看测试性能后调整。**若教师模型已在这108例GT上学习过，最终Test并不是真正独立于教师训练的测试集**，需追溯教师训练名单，并保留完全独立的GT患者作最终评价。生成器训练名单也必须审查。

## 4. 服务器上运行

先把完整Case ID清单CSV或原始UTF-8 txt粘贴文件保存在服务器私有目录，例如：
~~~text
/dataset/detection/detection_case_ids.csv
~~~

~~~bash
cd detection
git pull origin main
pip install -e '.[all]'

BASE=/dataset/downstream_detection
mkdir -p "$BASE/manifests"

liverdet build-cohorts \
  --case-list /dataset/detection/detection_case_ids.csv \
  --workdir "$BASE" \
  --labels-ap /dataset/detection/ap \
  --labels-pvp /dataset/detection/pvp \
  --val-fraction 0.5 \
  --seed 42
~~~

程序输出：
~~~text
/dataset/downstream_detection/manifests/
  cases.csv              # patient_id,phase,image_path,mask_path,ncct_path,split,label_source,cohort_type
  splits.csv             # patient_id,split,cohort_type；一患者一划分
  cohort_inventory.csv   # 全部输入记录，每期Image/Mask文件状态，包括待解决的缺失
~~~

如果伪标签仍缺失，可先核对 cohort_inventory.csv，确认后使用 --skip-missing-pseudo。Val/Test人工Mask缺失永远报错，不会通过 --skip-missing-pseudo 被跳过。

核对实际划分：
~~~bash
python - <<'PY'
import pandas as pd
base='/dataset/downstream_detection/manifests'
s = pd.read_csv(f'{base}/splits.csv', dtype={'patient_id':str})
print(s.groupby(['split','cohort_type']).size())
assert s.patient_id.is_unique
assert set(s.loc[s.split=='train','cohort_type']) == {'pseudo_label'}
assert set(s.loc[s.split!='train','cohort_type']) == {'ground_truth'}
PY
~~~

检查NIfTI几何并转换为2D：
~~~bash
liverdet audit --cases "$BASE/manifests/cases.csv" \
  --out "$BASE/manifests/nifti_audit.csv" --inspect-masks

liverdet prepare --cases "$BASE/manifests/cases.csv" \
  --splits "$BASE/manifests/splits.csv" --workdir "$BASE"
~~~

完整GT Mask的人工确认空层可用于Val/Test的阴性，伪Mask中教师未检出的空层不能自动视为训练阴性。Train如需负样本请提供额外的 --trusted-negatives.csv 人工审查列表。

五组实验（NCCT、REAL、SYN、REAL-AUG、REAL+SYN）使用同样拆分的Manifest。AP与PVP各自训练检测器，但Train/Val/Test患者一致。Val/Test是真实人工标签，不需要 --allow-pseudo-eval。

## 5. 防止标签泄漏

1. 同一个患者不能同时属于GT与伪标签集合，不能跨Train/Val/Test。
2. GT病例只用于Val/Test，严禁把同一批GT病例重新作为teacher监督训练后又报告为严格独立Test。
3. Val/Test中不得用教师伪标签替代缺失人工GT。
4. 不得把GT的病灶Mask输入被评估模型作为推理提示。
5. 保存完整cohort_inventory.csv、splits.csv、mask几何检查报告及各次模型训练配置。

## 6. 当前策略：只过滤极小的二维伪标签连通域（不使用Liver Mask）

`liverdet prepare` 默认不读取、不要求 AP/PVP Liver Mask，也不裁剪肝外伪标签。对 Train 伪标签按2D八连通域前景像素数量过滤，默认面积阈值4像素，由 `--min-component-pixels` 调整。Val/Test人工Tumor Mask不受任何面积阈值过滤。空伪标签切片不自动当作阴性。输出 `small_component_filter_qc.csv`（过滤前后组件数和阳性切片数）。

**处理过的数据建议写入新的workdir**，例如 `/dataset/downstream_detection_cc4`，避免误读以前可能保存的肝Mask过滤结果。详见 [SMALL_COMPONENT_FILTER.md](SMALL_COMPONENT_FILTER.md)。

## 7. 可选历史Liver Mask过滤（默认关闭）

仅当明确执行 `liverdet prepare --filter-pseudo-liver-mask` 时，才读取同期相 `ap_mask_registered.nii.gz` / `pvp_mask_registered.nii.gz` 做肝内ROI裁剪。对应审计增加 `liverdet audit --check-liver-mask`。**当前主线不启用**。旧方法详见 [LIVER_PSEUDO_FILTER.md](LIVER_PSEUDO_FILTER.md)。
