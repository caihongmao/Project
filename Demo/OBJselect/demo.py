from methods.func import *
from methods.camera import *

# camera = Camera_Realsense()
CAMERA = Camera_RGB(0)
GAZE = Gaze_Estimate(CAMERA)

def gaze_server():
    while True:
        # 获取关键帧
        CAMERA.get_image()
        canvas = CAMERA.image.copy()
        
        # 推理获得gaze
        GAZE.process_image(CAMERA.image, canvas)
        cv2.imshow('image', canvas)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    CAMERA.cap.release()

gaze_server()
