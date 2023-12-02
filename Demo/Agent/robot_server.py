from methods.func import *
from methods.camera import *
from methods.arduino import *
import requests

# camera = Camera_Realsense()
CAMERA = Camera_RGB(0)
ARDUINO = Arduino('COM7')
GAZE = Gaze_Estimate(CAMERA)

# API
url = {
    "robotstate" : "http://127.0.0.1:8010/robotstate"
    }

class robot_server:
    def __init__(self):
        # 激活态
        self.activate_window = []
        self.activate_state = False
        self.robotstate = 'none'

    def activate(self, gaze, gaze_pos):
        # 射线解算
        length_pixel = GAZE.detects_pre[0][0].bbox[1][1] - GAZE.detects_pre[0][0].bbox[0][1]
        d = 0.16 / (length_pixel / CAMERA.size[1]) / 0.18 * 0.25
        lh = 0.18 / 0.25 * d
        lw = 0.18 / 0.16 * d
        theta1 = np.arctan((gaze_pos[0] - 0.5)*lw / d) / np.pi * 180 - gaze[1] - 15
        theta0 = np.arctan((gaze_pos[1] - 0.5)*lh / d) / np.pi * 180 - gaze[0] - 5

        # 校准
        # print(theta0, theta1)

        # 激活判断
        if abs(theta1) <= 15 and abs(theta0) <= 15:
            self.activate_window = windowManage(self.activate_window, 1, 10)
        else:
            self.activate_window = windowManage(self.activate_window, 0, 10)
        
        if np.sum(self.activate_window) >= 7:
            self.activate_state = True

        elif np.sum(self.activate_window) <= 3:
            self.activate_state = False

    def voice(self):
        try:
            response = requests.get(url["robotstate"])
            state = response.json()['state']
            # 状态发生改变
            if self.robotstate != state:
                # 更新状态
                self.robotstate = state
                # arduino发生指令
                ARDUINO.arduino_exp(state)

        except:
            pass

    def servo(self, center):

        # 激活：控制舵机
        if self.activate_state:
            if center[0] > 0.55:
                if center[0] > 0.7:
                    ARDUINO.arduino_servo('right', 6)
                else:
                    ARDUINO.arduino_servo('right', 4)
                time.sleep(0.05)

            elif center[0] < 0.45:
                if center[0] < 0.3:
                    ARDUINO.arduino_servo('left', 6)
                else:
                    ARDUINO.arduino_servo('left', 4)
                time.sleep(0.05)

            if center[1] > 0.55:
                if center[1] > 0.7:
                    ARDUINO.arduino_servo('top', 6)
                else:
                    ARDUINO.arduino_servo('top', 4)
                time.sleep(0.05)

            elif center[1] < 0.45:
                if center[1] < 0.3:
                    ARDUINO.arduino_servo('down', 6)
                else:
                    ARDUINO.arduino_servo('down', 4)
                time.sleep(0.05)


        # 退激活：舵机复位
        else:
            if ARDUINO.servo_state[0] < 85:
                if ARDUINO.servo_state[0] < 75:
                    ARDUINO.arduino_servo('right', 6)
                else:
                    ARDUINO.arduino_servo('right', 4)
                time.sleep(0.05)

            elif ARDUINO.servo_state[0] > 95:
                if ARDUINO.servo_state[0] > 105:
                    ARDUINO.arduino_servo('left', 6)
                else:
                    ARDUINO.arduino_servo('left', 4)
                time.sleep(0.05)

            if ARDUINO.servo_state[1] > 95:
                if ARDUINO.servo_state[0] > 105:
                    ARDUINO.arduino_servo('down', 6)
                else:
                    ARDUINO.arduino_servo('down', 4)
                time.sleep(0.05)

            elif ARDUINO.servo_state[1] < 85:
                if ARDUINO.servo_state[1] < 75:
                    ARDUINO.arduino_servo('top', 6)
                else:
                    ARDUINO.arduino_servo('top', 4)
                time.sleep(0.05)

    def gaze(self, center):
        # 当前状态
        gaze_state = ARDUINO.gaze_state

        # 目标位置
        if self.activate_state:
            target_state = int((center[0] - 0.5) / (1 / 48)), int((center[1] - 0.5) / (1 / 48))
        else:
            target_state = (0, 0)
        
        # 纵向移动
        if gaze_state[1] > (target_state[1] + 3):
            ARDUINO.arduino_gaze('top', 3)
        elif gaze_state[1] > target_state[1]:
            ARDUINO.arduino_gaze('top', 1)
            time.sleep(0.01)
        elif gaze_state[1] < (target_state[1] - 3):
            ARDUINO.arduino_gaze('down', 3)
            time.sleep(0.01)
        elif gaze_state[1] < target_state[1]:
            ARDUINO.arduino_gaze('down', 1)
            time.sleep(0.01)

        # 横向移动
        if gaze_state[0] > (target_state[0] + 3):
            ARDUINO.arduino_gaze('left', 3)
            time.sleep(0.01)
        elif gaze_state[0] > target_state[0]:
            ARDUINO.arduino_gaze('left', 1)
            time.sleep(0.01)
        elif gaze_state[0] < (target_state[0] - 3):
            ARDUINO.arduino_gaze('right', 3)
            time.sleep(0.01)
        elif gaze_state[0] < target_state[0]:
            ARDUINO.arduino_gaze('right', 1)
            time.sleep(0.01)

    def run(self):
        count = 0
        while True:
            # 获取关键帧
            CAMERA.get_image()
            canvas = CAMERA.image.copy()

            # 推理获得gaze
            GAZE.process_image(CAMERA.image, canvas)

            # 非空
            if len(GAZE.detects_pre) > 0:
                # 激活判断
                gaze = GAZE.detects_pre[0][-1]
                gaze_pos = GAZE.detects_pre[0][0].landmarks[168] / CAMERA.size
                self.activate(gaze, gaze_pos)
                center = GAZE.detects_pre[0][1] / CAMERA.size

            if count < 4:
                count = count + 1
                # 对话状态
                # self.voice()
                
            else:
                count = 0
                # 非空
                if len(GAZE.detects_pre) > 0:
                    # 舵机动作
                    self.servo(center)
                    # 眼神动作
                    self.gaze(center)
                    
            cv2.imshow('image', canvas)
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
        cv2.destroyAllWindows()
        CAMERA.cap.release()

server = robot_server()
server.run()