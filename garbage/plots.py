# from ultralytics.utils.plotting import plot_results

# # 读取训练日志，生成曲线（保存到runs/detect/train/results.png）
# plot_results(file="./garbage_train/small_class_model12/results.txt")


from ultralytics import YOLO

# 加载当前训练的权重
model = YOLO("./garbage_train/small_class_model12/weights/best.pt")
# 跑验证集，自动生成混淆矩阵（保存到runs/detect/val/confusion_matrix.png）
model.val(data="garbage_small_class.yaml", plots=True)  # data替换为你的数据集配置文件