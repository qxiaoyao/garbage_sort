from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import cv2
import os
import asyncio
from pathlib import Path
from infer_and_map import infer_and_map_big_category

app = FastAPI(title="垃圾分类检测系统")

# 创建必要的目录
Path("uploads").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# 挂载静态文件和模板
app.mount("/results", StaticFiles(directory="results"), name="results")
templates = Jinja2Templates(directory="templates")

# 全局变量控制摄像头状态
camera_active = False
current_camera_source = 0


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """主页"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """上传图片并进行检测"""
    try:
        # 保存上传的图片
        upload_path = f"uploads/{file.filename}"
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 进行推理（注意返回值顺序：第一个是标注图片路径，第二个是检测结果）
        annotated_img_path, results = infer_and_map_big_category(upload_path, conf_thres=0.5)
        
        # 移动标注后的图片到results目录
        result_filename = f"result_{file.filename}"
        result_path = f"results/{result_filename}"
        
        # 检查标注图片是否存在并移动
        if os.path.exists(annotated_img_path):
            import shutil
            shutil.move(annotated_img_path, result_path)
        else:
            # 如果没有生成，复制原图
            import shutil
            shutil.copy(upload_path, result_path)
        
        # 整理检测结果，提取大分类统计
        category_count = {}
        for res in results:
            big_cat = res.get("大分类", "未知")
            small_cat = res.get("小分类", "未知")
            category_count[f"{big_cat}_{small_cat}"] = category_count.get(f"{big_cat}_{small_cat}", 0) + 1
        
        return {
            "success": True,
            "result_image": f"/results/{result_filename}",
            "categories": category_count,
            "details": results
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/camera/start/{source}")
async def start_camera(source: str):
    """启动摄像头"""
    global camera_active, current_camera_source
    
    try:
        # 解析摄像头源
        if source.isdigit():
            current_camera_source = int(source)
        else:
            # 支持IP摄像头地址 (例如: http://192.168.1.100:8080/video)
            current_camera_source = source
        
        camera_active = True
        return {"success": True, "message": f"摄像头 {source} 已启动"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/camera/stop")
async def stop_camera():
    """停止摄像头"""
    global camera_active
    camera_active = False
    return {"success": True, "message": "摄像头已停止"}


@app.get("/camera/stream")
async def video_stream():
    """视频流"""
    def generate():
        global camera_active, current_camera_source
        
        cap = cv2.VideoCapture(current_camera_source)
        
        if not cap.isOpened():
            print(f"无法打开摄像头: {current_camera_source}")
            return
        
        try:
            while camera_active:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取帧")
                    break
                
                # 保存帧为临时图片
                temp_img_path = "temp_frame.jpg"
                cv2.imwrite(temp_img_path, frame)
                
                # 推理+归类
                try:
                    annotated_img_path, results = infer_and_map_big_category(temp_img_path, conf_thres=0.5)
                    
                    # 读取标注后的图片（函数返回的路径）
                    if os.path.exists(annotated_img_path):
                        annotated_frame = cv2.imread(annotated_img_path)
                        # 清理标注图片
                        os.remove(annotated_img_path)
                    else:
                        annotated_frame = frame
                except Exception as e:
                    print(f"推理错误: {e}")
                    annotated_frame = frame
                
                # 编码为JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                # 生成MJPEG流
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        finally:
            cap.release()
            # 清理临时文件
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/camera/status")
async def camera_status():
    """获取摄像头状态"""
    global camera_active, current_camera_source
    return {
        "active": camera_active,
        "source": current_camera_source
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

