#!/bin/bash
# 串行跑剩余复现。可续跑(已出结果自动跳过)+ 治 OMP 崩溃 + 快的排前面。单个失败不阻断。
export KMP_DUPLICATE_LIB_OK=TRUE          # 治 OMP Error #15 (libiomp5md 双初始化)
export OMP_NUM_THREADS=4
PY="D:/anaconda/envs/minidector/python.exe"
cd "/c/Users/Windows11/Desktop/新版柑橘/复现_论文" || exit 1
run() {  # $1脚本 $2日志 $3说明 $4完成标志
  if [ -n "$4" ] && { [ -f "$4" ] || [ -d "$4" ]; }; then echo "===== 跳过(已完成): $3 ====="; return; fi
  echo "===== $3 =====  ($(date '+%H:%M:%S'))"
  "$PY" -u "$1" > "$2" 2>&1
  echo "  exit=$?  ($(date '+%H:%M:%S'))"
}
echo "[$(date '+%H:%M:%S')] START (可续跑)"

# 已完成会自动跳过: [28]基线 / [21]ViT
run train_citrusuat_yolov8cls.py  log_28_yolov8cls.txt    "[28] YOLOv8-cls 基线"    results_yolov8cls/citrusuat/results.csv
run train_vit_citrus.py           log_21_vit.txt          "[21] ViT 分类"           results_vit/vit_log.txt
# 快/有价值的排前面
run leafonlysam/run_leafonlysam.py log_53_leafonly.txt    "[53] Leaf-Only-SAM 抠叶" leafonlysam/leafmask_out
run simclr_pretrain.py            log_55_simclr.txt       "[55] SimCLR 自监督"      results_simclr/simclr_log.txt
# 自定义重建 + 检测(检测脚本用 ultralytics 默认多进程加载, 有 __main__ 保护)
run train_28_real.py              log_28_cms_real.txt     "[28] CMS 重建真方法"     results_yolov8cls/citrusuat_cms/results.csv
run train_scidb_yolov8det.py      log_36_det_baseline.txt "[36] YOLOv8 检测基线"    results_scidb_det/yolov8n/results.csv
run train_36_real.py              log_36_3ch_real.txt     "[36] 三通道 重建真方法"  results_scidb_det/yolov8n_3ch/results.csv

echo "[$(date '+%H:%M:%S')] DONE"
