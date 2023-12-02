from methods.gaze_estimator import *
from methods.filter import *
from methods.gaze import *
from methods.beamdetect import *
from common import Visualizer
from methods.utils import get_3d_face_model
from methods.func import *

def windowManage(window, n, length):
    if len(window) == length:
        window.pop(0)
    window.append(n)
    return window

def eura_distance(a, b):
    return np.linalg.norm(a - b)

def oor(x):
    if x > 1:
        return 1
    elif x < 0:
        return 0
    else:
        return x

class Gaze_Estimate:
    def __init__(self, camera):
        # 初始化
        self.gaze_estimator = GazeEstimator()
        self.camera = camera
        self.smooth_filter = KalmanFilter(4, 2, 0.001, 0.1)
        self.face_model_3d = get_3d_face_model()
        self.visualizer = Visualizer(self.gaze_estimator.camera,
                        self.face_model_3d.NOSE_INDEX)
        self.gaze_pos = None
        self.image = None
        self.faces = None
        self.detects_pre = []
        self.smooth_filters = []
        self.gazes = []
        
    def process_image(self, image, canvas) -> None:
        self.faces = self.gaze_estimator.detect_faces(image)

        # 获得所有脸部的中心点
        detects = []
        for face in self.faces:
            # 脸部追踪可视化
            cv2.rectangle(canvas, (face.bbox[0][0], face.bbox[0][1]), (face.bbox[1][0], face.bbox[1][1]), (0, 255, 0), 1)
            # 当前面部数据
            center = np.average(face.bbox,axis=0)
            detects.append([face, center])

        # 创建或继承卡尔曼滤波器
        if len(detects) > 0:
            # 如果上一帧脸为空，创建卡尔曼滤波器
            if len(self.detects_pre) == 0:
                for i in range(len(detects)):
                    detects[i].append(KalmanFilter(4, 2, 0.001, 0.1))
            else:
                # 根据距离匹配卡尔曼滤波器
                for i in range(len(detects)):
                    face_distance = []
                    for j in range(len(self.detects_pre)):
                        face_distance.append(eura_distance(detects[i][1], self.detects_pre[j][1]))
                    #如果小于某距离，继承滤波器 
                    if np.min(face_distance) < 50:
                        index = np.argmin(face_distance)
                        detects[i].append(self.detects_pre[index][2])
                    else:
                        detects[i].append(KalmanFilter(4, 2, 0.001, 0.1))

        self.gazes = []
        for i in range(len(detects)):
            face, center, kalman = detects[i]
            self.gaze_estimator.estimate_gaze(image, face)
            pitch, yaw = face.vector_to_angle(face.gaze_vector)
            kalman.process([pitch, yaw])
            gaze = kalman.state.reshape(-1)[:2]
            # 视线可视化
            draw_gaze(canvas, face.landmarks[168], gaze, length=200.0, thickness=2, color=(255, 0, 0))
            # 换算为角度
            gaze = gaze / np.pi * 180
            self.gazes.append(gaze)
            detects[i].append(gaze)

        self.detects_pre = detects
