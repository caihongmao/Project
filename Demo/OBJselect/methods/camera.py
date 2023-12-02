import cv2 as cv
from methods.realsense import *

class Camera_Realsense:
    def __init__(self):
        # 初始化RealSense相机
        self.pipeline, self.alignedFs = realsenseConfig()
        # 获取相机内参
        self.intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    def get_image(self):
        self.frames = self.pipeline.wait_for_frames()
        self.frames = self.alignedFs.process(self.frames)
        self.image, self.depth, self.ir = color_depth(self.frames)
        self.image = np.flip(self.image, axis=1)

class Camera_RGB:
    def __init__(self, index):  
        self.cap = cv.VideoCapture(index)
        self.image = False
        _, image = self.cap.read()
        self.size = image.shape[:2][::-1]

    def get_image(self):
        if not self.cap.isOpened():
            print("Cannot open camera")
        else:
            _, self.image = self.cap.read()
            self.image = np.flip(self.image, axis=1)
