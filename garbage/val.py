from ultralytics import  YOLO
# 加载最佳模型
best_model = YOLO('garbage_train/small_class_model12/weights/best.pt')

# 验证
val_results = best_model.val(
    data='garbage_small_class.yaml',
    imgsz=640,
    batch=16
)
print("小分类模型验证结果：", val_results.box.map50)  # 输出mAP@0.5