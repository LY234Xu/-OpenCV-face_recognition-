# 导入相关库
from picamera2 import Picamera2
import cv2
import numpy as np
import face_recognition
import os
import time

# 加载人脸图像并编码
def load_reference_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    if not face_locations:
        raise ValueError("No face detected in reference image")
    return face_recognition.face_encodings(rgb, face_locations)[0]

# 初始化摄像头
def init_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={
            "size": (640, 480),  # Camera resolution
            "format": "BGR888"
        },
        controls={
            "FrameDurationLimits": (33333, 66666),  # 30fps
            "AwbMode": 0,  # Auto white balance
            "ExposureTime": 20000  # 20ms exposure
        }
    )
    picam2.configure(config)
    picam2.start()
    return picam2

# 保存新的人脸图像并更新人脸编码数据
def save_new_face(frame, face_location, name):
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]
    # Convert BGR to RGB
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    # Create a directory to save face images if it doesn't exist
    if not os.path.exists('faces'):
        os.makedirs('faces')
    image_path = os.path.join('faces', f'{name}.jpg')
    cv2.imwrite(image_path, face_image_rgb)
    new_encoding = load_reference_image(image_path)
    return new_encoding

# 从人脸数据库中加载所有人脸
def load_all_known_faces():
    known_encodings = []
    known_names = []
    if os.path.exists('faces'):
        for filename in os.listdir('faces'):
            if filename.endswith('.jpg'):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join('faces', filename)
                try:
                    encoding = load_reference_image(image_path)
                    known_encodings.append(encoding)
                    known_names.append(name)
                except Exception as e:
                    print(f"Error loading {image_path}: {str(e)}")
    return known_encodings, known_names

# 主函数
def main():
    #让用户选择操作模式
    print("请选择操作模式：")
    print("1. 识别已有人脸数据")
    print("2. 录入人脸数据")
    choice = input("请输入选项 (1 或 2): ")

    # 加载已知人脸
    try:
        known_encodings, known_names = load_all_known_faces()
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        return

    # 初始化相机
    camera = init_camera()

    # 显示参数
    SCALE_FACTOR = 0.5
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    THRESHOLD = 0.5

    # 计时器设置
    TIMEOUT = 3  # 等待询问保存的时间（秒）
    unknown_start_time = None

    try:
        while True:
            # 捕获帧
            frame = camera.capture_array()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 预处理
            small_frame = cv2.resize(
                frame_rgb,
                (0, 0),
                fx=SCALE_FACTOR,
                fy=SCALE_FACTOR,
                interpolation=cv2.INTER_AREA
            )

            # 人脸检测
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            # 识别处理
            all_unknown = True
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # 恢复原始坐标
                top = int(top / SCALE_FACTOR)
                right = int(right / SCALE_FACTOR)
                bottom = int(bottom / SCALE_FACTOR)
                left = int(left / SCALE_FACTOR)

                # 计算匹配距离
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                min_distance = np.min(distances)
                match_index = np.argmin(distances)

                # 确定身份
                name = "Unknown"
                color = (0, 0, 255)  # 红色
                if min_distance <= THRESHOLD:
                    name = known_names[match_index]
                    color = (0, 255, 0)  # 绿色
                    all_unknown = False

                # 绘制边界框和标签
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                text = f"{name} ({min_distance:.2f})"
                cv2.putText(frame, text, (left + 6, bottom - 6),
                            FONT, 0.5, color, 1)

            # 处理未知人脸
            if choice == '2' and all_unknown:
                if unknown_start_time is None:
                    unknown_start_time = time.time()
                elif time.time() - unknown_start_time >= TIMEOUT:
                    # 提示用户保存新人脸
                    save_choice = input("检测到未知人脸一段时间。是否保存此人脸? (y/n): ")
                    if save_choice.lower() == 'y':
                        new_name = input("请输入此人姓名: ")
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            top = int(top / SCALE_FACTOR)
                            right = int(right / SCALE_FACTOR)
                            bottom = int(bottom / SCALE_FACTOR)
                            left = int(left / SCALE_FACTOR)
                            new_encoding = save_new_face(frame, (top, right, bottom, left), new_name)
                            known_encodings.append(new_encoding)
                            known_names.append(new_name)
                            print(f"{new_name} 的人脸已保存。")
                    unknown_start_time = None
            else:
                unknown_start_time = None

            # 显示输出
            cv2.imshow('Face Recognition', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # 退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户终止程序")
                break

    finally:
        # 清理资源
        camera.stop()
        camera.close()
        cv2.destroyAllWindows()
        print("系统资源已释放")

if __name__ == "__main__":
    main()
