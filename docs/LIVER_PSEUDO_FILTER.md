# 历史可选方案：Liver ROI 伪标签过滤（当前默认不启用）

**最新主线已切换为仅过滤极小2D伪标签连通域，不做Liver Mask裁剪。** 详见 [SMALL_COMPONENT_FILTER.md](SMALL_COMPONENT_FILTER.md)。

如需复现本文以下旧方法，必须明确添加 `liverdet prepare --filter-pseudo-liver-mask` 及 `liverdet audit --check-liver-mask`，并写入新的独立workdir。下文旧命令缺少这些参数，直接执行不会触发肝Mask过滤。

# 伪标签肝脏区域过滤：Liver ROI Filtering

## 1. 固定文件路径

    /dataset/dataset/gx_all_registered/A0_1004/ap_registered.nii.gz
    /dataset/dataset/gx_all_registered/A0_1004/ap_mask_registered.nii.gz      # AP liver mask
    /dataset/dataset/gx_all_registered/A0_1004/pvp_registered.nii.gz
    /dataset/dataset/gx_all_registered/A0_1004/pvp_mask_registered.nii.gz     # PVP liver mask
    /dataset/detection/ap/A0_1004.nii.gz                                    # AP lesion pseudo mask
    /dataset/detection/pvp/A0_1004.nii.gz                                   # PVP lesion pseudo mask

同一患者的 liver mask 在注册影像目录内；外部检测伪标签在 /dataset/detection/ap 或 pvp 内。Liver Mask 不是肿瘤标注，不参与生成病灶框，专用于过滤伪标签。

## 2. 处理规则

1. Train 且 label_source=pseudo：读取外部伪标签与相同期相liver mask，先检查shape/affine与真实增强影像一致，再在体素空间取交集：filtered_lesion = (pseudo_mask > 0) & (liver_mask > 0)。
2. 逐切片在交集上提取连通区域的2D BBox，过滤肝外伪阳性。
3. Val/Test 且 label_source=manual：不做任何liver mask过滤，保留原始人工肿瘤真值，包括可能伸出肝边缘的病灶。
4. 过滤后伪标签为空：跳过该切片，不自动将其当作阴性。只有经过审查并明确记录的可信阴性才能作为训练负样本。
5. Liver Mask缺失、空间几何不一致或有非有限值：中止并报告。不能静默改用未过滤伪标签。
6. 对跨越肝边界的伪预测，只保留肝内体素；重新生成矩形BBox时仍可能包含少量肝外背景。

## 3. 源码位置

- src/liverdet/prepare.py：过滤并提取BBox，仅对pseudo标签启用。
- src/liverdet/audit.py：预检查同一期相Liver Mask，统计肝外伪标签体素。
- tests/test_liver_roi_filter.py：覆盖肝内保留、肝外删除、GT不受影响、缺失/错位Mask报错。

不改动患者级splits、NCCT匹配、五组数据导出或三种检测器。

## 4. 服务器执行

    cd detection
    git pull origin main
    pip install -e '.[all]'

    BASE=/dataset/downstream_detection

    liverdet build-cohorts \
      --case-list /dataset/detection/detection_case_ids.csv \
      --workdir "$BASE" \
      --labels-ap /dataset/detection/ap \
      --labels-pvp /dataset/detection/pvp \
      --val-fraction 0.5 --seed 42

    liverdet audit \
      --cases "$BASE/manifests/cases.csv" \
      --out "$BASE/manifests/nifti_audit.csv" \
      --inspect-masks

    liverdet prepare \
      --cases "$BASE/manifests/cases.csv" \
      --splits "$BASE/manifests/splits.csv" \
      --workdir "$BASE"

## 5. 质控输出

新增：/dataset/downstream_detection/manifests/liver_filter_qc.csv

包含：patient_id、phase、split、pseudo_mask_path、liver_mask_path、pseudo_voxels_before、pseudo_voxels_inside_liver、pseudo_voxels_removed、positive_slices_before、positive_slices_after。

audit输出亦包含：liver_mask_path、liver_shape、liver_affine_max_abs_diff、pseudo_voxels_inside_liver、pseudo_voxels_outside_liver。

统计结果用于核对过滤是否错删边缘病灶。Liver Mask本身可能漏分割，请对大量删除体素的病例进行人工质控。原始NIfTI及伪标签不被修改。

## 6. 与主线五组实验关系

- Train：Pseudo Mask × Liver Mask → 2D病灶框 → REAL/SYN/REAL-AUG/REAL+SYN训练。
- Val/Test：人工Tumor Mask → 原始2D框（金标准，不过滤）。
- NCCT：仍需正确匹配与审查空间对应关系。此次更改只限制训练伪标签，不自动修复NCCT错层。
- 该处理是训练标签过滤，并非推理阶段的检测后处理。模型预测本身未受到肝脏Mask约束。
