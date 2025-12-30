from ultralytics import YOLO

# 加载预训练模型
model = YOLO('./garbage_train/small_class_model12/weights/best.pt')  # 可选：yolov11s.pt/yolov11m.pt（更大模型精度更高）

# 启动训练（关键参数说明）
results = model.train(
    data='garbage_small_class.yaml',  # 步骤1.3配置的yaml文件
    epochs=40,                       # 训练轮次（根据数据集大小调整，小数据集50-80即可）
    batch=64,                         # 批次大小（根据GPU显存调整，显存不足改8/4）
    imgsz=640,                        # 输入图片尺寸（640/800，越大精度越高）
    # lr0=0.05,                         # 初始学习率（默认即可）
    device=0,                         # GPU编号（0为单GPU，多GPU写[0,1]）
    patience=30,                      # 早停耐心值（30轮无提升停止训练）
    save=True,                        # 保存最佳模型
    project='garbage_train',          # 训练结果保存路径
    name='small_class_model',         # 模型名称
    resume=True,
    # pretrained=True,                  # 使用预训练权重（迁移学习）
    augment=True                      # 数据增强（提升泛化能力）
)
7
# 训练完成后，最佳模型保存在：garbage_train/small_class_model/weights/best.pt