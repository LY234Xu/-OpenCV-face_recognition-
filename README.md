# 基于OpenCV和face_recognition库实现的人脸识别

## 配置环境

### OpenCV库

*     pip install opencv -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
  

### face_recognition库

* 在安装face_recognition库前要先安装cmake库、dlib库
  
  * cmake库可以直接下载
    
        pip install cmake -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    
  * dlib库要先下载对应Python版本的whl文件下载路径：[Python3.12版本whl文件下载]([GitHub - Silufer/dlib-python: Dlib compiled wheels for Python on Windows X64](https://github.com/Silufer/dlib-python/tree/main))下载后放在某个文件夹并记住路径(例：直接存在D盘中)
    
        pip install D:\dlib-19.24.2-cp312-cp312-win_amd64.whl
    
    参考资料：[dlib库安装教程]([[已解决]face_recognition库安装，dlib库安装-CSDN博客](https://blog.csdn.net/weixin_53236070/article/details/124306424))
    
  * cmake库和dlib库安装完成后即可安装face_recogniton库
    
        pip install face_recognition -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    
  * 配置完成
    

### 将本地库链接到Pycharm

* 找到通过`pip`安装到本地的库的地址
  
      C:\Users\LENOVO>pip show dlib
      Name: dlib
      Version: 19.24.2
      Summary: A toolkit for making real world machine learning and data analysis applications
      Home-page: https://github.com/davisking/dlib
      Author: Davis King
      Author-email: davis@dlib.net
      License: Boost Software License
      Location: D:\Anaconda\Lib\site-packages
      Requires:
      Required-by: face-recognition
  
  **Location就是本地库的地址：** Location: D:\Anaconda\Lib\site-packages
  
* **Pycharm：**：文件 ==> 设置 ==> 项目：*项目名* ==> 项目结构 ==> 添加内容根 ==> 将刚才得到的本地库地址添加进去 ==> 确定
  
* 进入Pycharm尝试导入本地库，看是否报错
  
* **参考资料：**[pycharm使用本地已安装的第三方库]([pycharm使用本地已安装的第三方库_python中已经安装了人第三方库在pycharm中怎么用-CSDN博客](https://blog.csdn.net/qq_43650934/article/details/103395770))
  

* * *

## 项目流程

### 特征图像准备

* 准备人脸图像，作为数据库的原始素材
  
* 放入项目根目录中，一般专门创建一个目录保存素材
  

### 导入库

    import cv2
    import numpy as np
    import face_recognition

### 加载图片

    xu = cv2.imread('xu yinan.jpg')
    lin = cv2.imread('lin yaojian.jpg')
    chen = cv2.imread('chen mingjie.jpg')

### BGR转RGB（使用OpenCV内置方法转换颜色空间，优于数组切片）

* **语法格式：dst = cv2.cvtColor(src,code,dstCn)**
  
  * dst表示输出图像，与原始输出图像具有同样的数据类型和深度
    
  * src表示原始输入图像
    
  * code是色彩空间转换码：如下图![](file:///C:/Users/LENOVO/Desktop/%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4%E8%BD%AC%E6%8D%A2%E7%A0%81.png?msec=1742653141879)
    
  * dstCn是目标图像的通道数。如果参数为默认的0，则通道数自动通过原始输入图像和code得到
    

### 检测人脸并提取特征编码（添加错误处理）

* **人脸检测函数：** face_recognition.face_locations(image)
  
  * image是RGB格式的原始图像
    
  * 函数会自动识别image中的人脸并返回一个列表
    
  * 此处是作数据库，所有素材应为只有一个一张人脸，所以只返回列表的第一个元素`[0]`给`xu_face`
    
* **提取特征编码函数：** face_recognition.face_encodings(image,[locations])
  
  * image是RGB格式的原始图像
    
  * locations是经过人脸检测后的位置数据
    
  * 函数根据原始图像和人脸位置数据返回编码值列表。因为这里只有一个人脸数据，所以直接返回第一个元素即可`[0]`
    
* **添加错误处理：** 若未检测到人脸，则返回错误信息
  
      try:
          xu_face = face_recognition.face_locations(xu_RGB)[0]  # 假设只取第一张检测到的人脸
          xu_encoding = face_recognition.face_encodings(xu_RGB, [xu_face])[0]
      
          lin_face = face_recognition.face_locations(lin_RGB)[0]
          lin_encoding = face_recognition.face_encodings(lin_RGB, [lin_face])[0]
      
          chen_face = face_recognition.face_locations(chen_RGB)[0]
          chen_encoding = face_recognition.face_encodings(chen_RGB, [chen_face])[0]
      
      except IndexError:
          raise Exception("未在参考图片中检测到人脸，请检查图片质量或人脸位置")
  

### 创建数据库

* 将编码好的人脸特征数据保存为一个列表
  
* 保存对应人脸特征的姓名
  
      known_encodings = [xu_encoding, lin_encoding,chen_encoding]
      known_names = ['xu yinan', 'lin yaojian','chen mingjie']
  

### 视频流处理（摄像头开启判断）

#### 摄像头初始化

* cv2.VideoCaptrue(source)
  
* **参数：** source
  
  * **摄像头设备索引：** 整数类型（如：`0`表示默认摄像头，`1` 表示第二个摄像头）
    
  * **视频文件路径：** 字符串类型（如：“D:\video.mp4”）
    
  * **网络流URL：** 字符串类型（如：“rtsp://example.com/steam”）
    
* 判断是否成功开启摄像头
  
      cap = cv2.VideoCapture(0)
      if not cap.isOpened():
          raise IOError('无法打开摄像头，请检查设备连接')
  

#### 配置显示参数

    SCALE_FACTOR = 0.5  # 图像缩放因子
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_COLOR = (255, 255, 255)
    BOX_COLOR = (0, 255, 0)
    THICKNESS = 2
    THRESHOLD = 0.5  # 人脸匹配阈值（可根据实际调整）

* **SCALE_FACTOR:** 图像缩放因子，用于在图像处理前按比例缩小图像尺寸
  
* **FONT = cv2.FONT_HERSHEY_SIMPLEX:** OpenCV中预定义的字体类型，表示使用`HERSHEY_SIMPLEX` 风格字体
  
* **FONT_SCALE:** 字体大小的缩放比例，相对于基准字体尺寸（1.0）进行缩放
  
* **FONT_COLOR:** 字体颜色，此处为白色
  
* **BOX_COLOR:** 边界框颜色，此处为绿色
  
* **THICKNESS:** 线条粗细（单位：像素），用于控制边界框或文字的线条宽度
  
* **THRESHOLD:** 置信度阈值，用于过滤低置信度的检测结果
  

#### 获取视频流

    while True:
        ret,frame = cap.read()
        if not ret:
            print("无法获取视频帧，正在退出...")
            break

* **ret:** 布尔类型，用于判断是否成功获取视频帧
  
* **frame：** 获取的视频帧，只是一帧，所以要用循环不断获取视频帧构成视频
  

#### 图像预处理

     small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

* **参数解析：**
  
  * **`frame`**  原始输入图像（NumPy 数组格式，通常由 `cv2.imread()` 或 `cap.read()` 读取
    
  * **`(0,0)`** 目标图像尺寸 `(width, height)`，设为 `(0, 0)` 表示**不直接指定尺寸**，而是通过 `fx` 和 `fy` 缩放因子计算
    
  * **`fx=SCALE_FACTOR`** 水平方向（宽度）的缩放因子，例如 `0.5` 表示宽度缩小到原图的 50%
    
  * **`fy=SCALE_FACTOR`** 垂直方向（高度）的缩放因子，与 `fx` 一致时保持宽高比不变
    
* 第二行就是将捕获到的每一帧图片也转换成`RGB` 格式便于后期人脸识别操作
  

### 对视频帧进行人脸检测和识别

     face_locations = face_recognition.face_locations(rgb_small_frame)
     face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

与素材图像的操作同理，对每一视频帧进行人脸位置检测，并对位置数据进行编码

### 遍历每一个检测到的人脸

     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

* [zip()函数作用]([腾讯元宝 - 轻松工作 多点生活](https://yuanbao.tencent.com/bot/app/share/chat/8yTO2FS6TmKU))
  
* **补充：** face_locations是一个列表，格式为（top,right,bottom,left）
  
* **还原坐标到原始尺寸**
  
      top = int(top / SCALE_FACTOR)
      right = int(right / SCALE_FACTOR)
      bottom = int(bottom / SCALE_FACTOR)
      left = int(left / SCALE_FACTOR)
  

### 与数据库中的人脸进行匹配

    distances = face_recognition.face_distance(known_encodings, face_encoding)
    min_distance = np.min(distances)
    min_index = np.argmin(distances)

* ### **`distances = face_recognition.face_distance(known_encodings, face_encoding)`**
  
  * ​**功能**：计算当前检测到的人脸编码（`face_encoding`）与已知人脸编码库（`known_encodings`）中每个人脸的特征距离。
  * ​**参数**：
    * `known_encodings`：已知人脸的编码列表（例如：数据库存储的人脸特征向量）。
    * `face_encoding`：当前检测到的人脸编码（单个人脸的特征向量）。
  * ​**返回值**：`distances` 是一个数组，表示当前人脸与每个已知人脸的相似度距离（值越小越相似）。
  * ​**底层原理**：通常使用**欧氏距离**​（Euclidean Distance）或**余弦相似度**​（Cosine Similarity）计算。
* ### **`min_distance = np.min(distances)`**
  
  * ​**功能**：从 `distances` 数组中提取最小的距离值。
  * ​**意义**：表示当前人脸与已知人脸库中**最接近的匹配项**的相似度。
* ### **`min_index = np.argmin(distances)`**
  
  * ​**功能**：从 `distances` 数组中提取最小距离值的**索引位置**。
  * ​**意义**：通过索引（`min_index`）可以找到对应的已知人脸身份（例如：从名称列表 `known_names` 中获取 `known_names[min_index]`）。

### 确定身份（添加阈值判断）

* ​**阈值（`THRESHOLD`）​**：需根据实际场景调整：
  
  * ​**严格阈值**​（如 `0.4`）：减少误报，但可能漏检。
  * ​**宽松阈值**​（如 `0.6`）：增加识别率，但可能引入误报。
* ​**示例阈值参考**：
  
  * 欧氏距离通常以 `0.6` 为分界点（值越小越相似）。
  * 余弦相似度通常以 `0.8` 以上为匹配（值越大越相似）。
  
      name = "Unknown"
      if min_distance <= THRESHOLD:
          name = known_names[min_index]
  

### 绘制结果

    cv2.rectangle(frame, (left, top), (right, bottom), BOX_COLOR, THICKNESS)
    
    text_size = cv2.getTextSize(name, FONT, FONT_SCALE, THICKNESS)[0]
    cv2.rectangle(frame,(left, bottom - text_size[1] - 10),(left + text_size[0] + 20, bottom),BOX_COLOR, cv2.FILLED)
    cv2.putText(frame, name,(left + 10, bottom - 10),FONT, FONT_SCALE, FONT_COLOR, THICKNESS)

### 显示结果

    cv2.imshow('Face Recognition System', frame)

### 退出机制

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("用户主动终止程序")
        break

### 资源释放

    cap.release()
    cv2.destroyAllWindows()
    print("系统资源已释放，程序正常退出")

## 完整代码

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
