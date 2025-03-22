# 1、加载库（移除了无用的astropy导入）
import cv2
import numpy as np
import face_recognition

# 2、加载图片
xu = cv2.imread('xu yinan.jpg')
lin = cv2.imread('lin yaojian.jpg')
chen = cv2.imread('chen mingjie.jpg')

# 3、BGR转RGB（优化了数组切片操作）
xu_RGB = cv2.cvtColor(xu, cv2.COLOR_BGR2RGB)
lin_RGB = cv2.cvtColor(lin, cv2.COLOR_BGR2RGB)
chen_RGB = cv2.cvtColor(chen, cv2.COLOR_BGR2RGB)

# 4、检测人脸并提取特征编码（添加错误处理）
try:
    xu_face = face_recognition.face_locations(xu_RGB)[0]  # 假设只取第一张检测到的人脸
    xu_encoding = face_recognition.face_encodings(xu_RGB, [xu_face])[0]

    lin_face = face_recognition.face_locations(lin_RGB)[0]
    lin_encoding = face_recognition.face_encodings(lin_RGB, [lin_face])[0]

    chen_face = face_recognition.face_locations(chen_RGB)[0]
    chen_encoding = face_recognition.face_encodings(chen_RGB, [chen_face])[0]

except IndexError:
    raise Exception("未在参考图片中检测到人脸，请检查图片质量或人脸位置")

# 5、创建数据库（确保编码结构正确）
known_encodings = [xu_encoding, lin_encoding,chen_encoding]
known_names = ['xu yinan', 'lin yaojian','chen mingjie']

# 6、视频流处理（优化了摄像头初始化）
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('无法打开摄像头，请检查设备连接')

# 配置显示参数
SCALE_FACTOR = 0.5  # 图像缩放因子
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 255, 0)
THICKNESS = 2
THRESHOLD = 0.5  # 人脸匹配阈值（可根据实际调整）

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频帧，正在退出...")
        break

    # 7、图像预处理
    small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 8、人脸检测与识别
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # 遍历每个检测到的人脸
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 还原坐标到原始尺寸
        top = int(top / SCALE_FACTOR)
        right = int(right / SCALE_FACTOR)
        bottom = int(bottom / SCALE_FACTOR)
        left = int(left / SCALE_FACTOR)

        # 9、人脸匹配（优化匹配逻辑）
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        min_distance = np.min(distances)
        min_index = np.argmin(distances)

        # 10、确定身份（添加阈值判断）
        name = "Unknown"
        if min_distance <= THRESHOLD:
            name = known_names[min_index]

        # 11、绘制结果（优化显示位置）
        cv2.rectangle(frame, (left, top), (right, bottom), BOX_COLOR, THICKNESS)
        text_size = cv2.getTextSize(name, FONT, FONT_SCALE, THICKNESS)[0]
        cv2.rectangle(frame,
                      (left, bottom - text_size[1] - 10),
                      (left + text_size[0] + 20, bottom),
                      BOX_COLOR, cv2.FILLED)
        cv2.putText(frame, name,
                    (left + 10, bottom - 10),
                    FONT, FONT_SCALE, FONT_COLOR, THICKNESS)

    # 12、显示结果
    cv2.imshow('Face Recognition System', frame)

    # 退出机制（添加了退出提示）
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("用户主动终止程序")
        break

# 13、资源释放（添加了更全面的清理）
cap.release()
cv2.destroyAllWindows()
print("系统资源已释放，程序正常退出")