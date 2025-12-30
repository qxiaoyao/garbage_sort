import cv2
from garbage_sorting.infer_and_map import infer_and_map_big_category
def realtime_infer_and_map(video_path=1):  # 0为摄像头，也可传入视频文件路径
    cap = cv2.VideoCapture(video_path,cv2.CAP_MSMF)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        # 保存帧为临时图片（或直接用YOLO推理帧）
        temp_img_path = "./temp_frame.jpg"
        cv2.imwrite(temp_img_path, frame)

        # 推理+归类
        _, results = infer_and_map_big_category(temp_img_path, conf_thres=0.5)

        # 显示结果
        annotated_frame = cv2.imread("./annotated_temp_frame.jpg")
        cv2.imshow("Garbage Big Category Detection", annotated_frame)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 调用实时推理
realtime_infer_and_map(2)  # 摄像头实时检测