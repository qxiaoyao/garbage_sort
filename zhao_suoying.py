import cv2
import sys


def find_camera_index():
    """
    修复DSHOW警告：改用MSMF后端检测摄像头，兼容虚拟摄像头（iVCam）
    返回：可用摄像头索引+设备信息
    """
    available_cameras = []
    # 遍历前10个索引（覆盖绝大多数场景）
    for idx in range(10):
        # 方案1：改用MSMF后端（推荐，Windows虚拟摄像头优先）
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        # 方案2：不指定后端（让OpenCV自动选择，兜底）
        # cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            # 获取摄像头基础信息（区分iVCam）
            try:
                # 获取摄像头分辨率（辅助区分）
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cam_info = f"分辨率({width}x{height})_MSMF"
                # 尝试读取一帧（验证是否为iVCam）
                ret, frame = cap.read()
                if ret and "iVCam" in cap.getBackendName():  # 若能识别名称
                    cam_info = "e2eSoft iVCam（手机摄像头）"
            except Exception as e:
                cam_info = f"未知设备_索引{idx}"

            available_cameras.append((idx, cam_info))
            cap.release()  # 释放摄像头
        else:
            continue

    # 若未找到，尝试禁用后端强制自动选择
    if not available_cameras:
        print("⚠️ MSMF后端未找到摄像头，尝试自动选择后端...")
        for idx in range(10):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                available_cameras.append((idx, f"自动识别_索引{idx}"))
                cap.release()
    return available_cameras


# 执行检测
if __name__ == "__main__":
    cameras = find_camera_index()
    if cameras:
        print("✅ 找到可用摄像头：")
        for idx, info in cameras:
            print(f"   索引 {idx} → {info}")
    else:
        print("❌ 未找到任何可用摄像头")
        sys.exit(1)