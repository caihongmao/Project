import serial
import time

class Arduino:
    def __init__(self,  portName):
        self.portName = portName
        self.baudRate = 9600
        self.timeOut = 1
        self.ser = None

        # 动作状态
        self.servo_state = [90, 90]
        self.gaze_state = [0, 0]
        self.arduino_init()
        time.sleep(3)

    # 初始化arduino
    def arduino_init(self):
        try:
            self.ser = serial.Serial(self.portName, self.baudRate, timeout=self.timeOut)
            print("连接到串口")
        except serial.SerialException as e:
            print(f"无法连接到串口: {e}")
            exit()

    def arduino_servo(self, param, degree):
        if param == 'left':
            instruct = '0 0 ' + str(max(self.servo_state[0] - degree, 0)) + '\n'
            # 更新
            self.servo_state[0] = max(self.servo_state[0] - degree, 0)

        elif param == 'right':
            instruct = '0 0 ' + str(min(self.servo_state[0] + degree, 180)) + '\n'
            # 更新
            self.servo_state[0] = min(self.servo_state[0] + degree, 180)

        elif param == 'top':
            instruct = '0 1 ' + str(min(self.servo_state[1] + degree, 180)) + '\n'
            # 更新
            self.servo_state[1] = min(self.servo_state[1] + degree, 180)

        elif param == 'down':
            instruct = '0 1 ' + str(max(self.servo_state[1] - degree, 0)) + '\n'
            # 更新
            self.servo_state[1] = max(self.servo_state[1] - degree, 0)

        print('指令:' + str(instruct)[:-1])
        self.ser.write(instruct.encode())
        return
    
    def arduino_exp(self, param):
        if param == 'none':
            instruct = '2 0\n'

        elif param == 'listen':
            instruct = '2 0\n'

        elif param == 'think':
            instruct = '2 1\n'

        elif param == 'speak':
            instruct = '2 2\n'

        print('指令:' + str(instruct)[:-1])
        self.ser.write(instruct.encode())
        return


    def arduino_gaze(self, param, degree=1):
        if param == 'left':
            instruct = '1 ' + str(max(self.gaze_state[0] - degree, -12)) +' ' +str(self.gaze_state[1]) + '\n'
            # 更新
            self.gaze_state[0] = max(self.gaze_state[0] - degree, -12)

        elif param == 'right':
            instruct = '1 ' + str(min(self.gaze_state[0] + degree, 12)) +' ' + str(self.gaze_state[1]) + '\n'
            # 更新
            self.gaze_state[0] = min(self.gaze_state[0] + degree, 12)

        elif param == 'down':
            instruct = '1 ' + str(self.gaze_state[0]) +' ' + str(min(self.gaze_state[1] + degree, 12)) + '\n'
            # 更新
            self.gaze_state[1] = min(self.gaze_state[1] + degree, 12)

        elif param == 'top':
            instruct = '1 ' + str(self.gaze_state[0]) + ' ' + str(max(self.gaze_state[1] - degree, -12)) + '\n'
            # 更新
            self.gaze_state[1] = max(self.gaze_state[1] - degree, -12)

        # print('指令:' + str(instruct)[:-1])
        self.ser.write(instruct.encode())
        return
        
