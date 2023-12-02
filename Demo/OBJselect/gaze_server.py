from methods.func import *
from methods.camera import *
from methods.beamdetect import *
CAMERA = Camera_RGB(0)
GAZE = Gaze_Estimate(CAMERA)

class Obj_Dict:
    def __init__(self):
        self.sphere_list = {
            'a':[-0.500, -0.060, -0.170, 0.2],
            'b':[0.500, -0.060, -0.170, 0.2],
            'e':[0.0, -0.84, 0.3, 0.3],
            'f':[0.0, -0.11, -0.06, 0.3],
            'c':[-0.80, -0.7, -0.12, 0.4],
            'd':[0.80, -0.7, -0.12, 0.4],
        }

        self.rectangle_list = {
        }

obj = Obj_Dict()

class Gaze_Server:
    def __init__(self):
        self.faceid = []

    def coord(self, gaze_pos):
        length_pixel = GAZE.detects_pre[0][0].bbox[1][1] - GAZE.detects_pre[0][0].bbox[0][1]
        d = 0.16 / (length_pixel / CAMERA.size[1]) / 0.18 * 0.25
        lw = 0.18 / 0.18 * d
        lh = 0.21 / 0.20 * d
        x = (gaze_pos[0] - 0.5) *lw
        y = (gaze_pos[1] - 0.5) *lh
        return np.array([x, y, -d])

    def process(self):
        while True:
            # 获取关键帧
            CAMERA.get_image()
            canvas = CAMERA.image.copy()
            # 推理获得gaze
            GAZE.process_image(CAMERA.image, canvas)

            # 多于一人
            if len(GAZE.faces) > 0:
                for i in range(len(GAZE.detects_pre)):
                    # 视线
                    gaze = GAZE.detects_pre[i][-1]
                    # 位置
                    gaze_pos = GAZE.detects_pre[i][0].landmarks[168] / CAMERA.size
                    # 中心点
                    center = GAZE.detects_pre[i][1] / CAMERA.size
                    # 坐标
                    # print(self.coord(gaze_pos))
                    gaze_vector = pitchyaw_to_vector(np.array([gaze/180 * 3.14]) )[0]
                    print(gaze_vector)
                
                    # 姿态
                    # print(gaze_vector)
                    image, select_state = beam_obj(self.coord(gaze_pos), gaze_vector, obj.sphere_list, obj.rectangle_list)
                    cv2.imshow('windw', image)
                    

            cv2.imshow('image', canvas)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
        CAMERA.cap.release()

gaze_server = Gaze_Server()
gaze_server.process()