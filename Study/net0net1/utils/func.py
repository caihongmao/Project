import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# realsense设置
def color_depth(frames):
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_frame
            
def realsenseConfig():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    # 对齐设置
    align_to = rs.stream.color
    alignedFs = rs.align(align_to)
    pipeline.start(config)
    return pipeline, alignedFs

# 获取世界坐标
def getworldCoord(x, y, depth_frame, intrinsics):
    depth_value = depth_frame.get_distance(x, y)
    camera_point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_value)
    return camera_point

# 可视化方法
def draw(results, canvas):
    if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=canvas,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
    return canvas

def cv2text(canvas,name, d, y):
    cv2.putText(canvas, text= name + ':{}'.format(d),\
                org=(10, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
                fontScale=0.8, thickness=2, lineType=cv2.LINE_AA, color=(0, 0, 255))

# basic methods 
def dic2array(data):
    return np.array([[i.x, i.y, i.z] for i in data])


def get_image(frame, array):
    left_array = np.array([array[173],  array[130], array[470], array[23]]).astype('int')
    right_array = np.array([array[463], array[359], array[253], array[258]]).astype('int')
    # xmin, xmax, ymin, ymax
    left_box = [min(left_array[:,:1])[0], max(left_array[:,:1])[0], min(left_array[:,1:2])[0], max(left_array[:,1:2])[0]]
    right_box = [min(right_array[:,:1])[0], max(right_array[:,:1])[0], min(right_array[:,1:2])[0], max(right_array[:,1:2])[0]]
    left_box = resize_box(left_box)
    right_box = resize_box(right_box)
    left_img = frame[left_box[2]: left_box[3], left_box[0]:left_box[1]]
    right_img = frame[right_box[2]: right_box[3], right_box[0]:right_box[1]]
    return left_img, right_img

def resize_box(box):
    if box[1] - box[0] >= (box[3] - box[2]) * 60 / 36:
        center = (box[3] + box[2]) / 2
        box[2] =  int(center - (box[1] - box[0]) / 60 * 18)
        box[3] =  int(center + (box[1] - box[0]) / 60 * 18)
    else:
        center = (box[0] + box[1]) / 2
        box[0] = int(center -(box[3] - box[2]) / 36 * 30)
        box[1] = int(center +(box[3] - box[2]) / 36 * 30)
    return box

def get_eye_world_coord(image_points, canvas, depth, intrinsics):
    image_points_flip = image_points.copy()
    image_points_flip[:,:1] = 1280 - image_points[:,:1]
    x0 = int(image_points_flip[468][0])
    y0 = int(image_points_flip[468][1])
    x1 = int(image_points_flip[473][0])
    y1 = int(image_points_flip[473][1])
    worldCoord = []
    if x0 > 0 and x0 < 1280 and y0 > 0 and y0 < 720:
        worldCoord.append(rs.rs2_deproject_pixel_to_point(intrinsics, [x0, y0], depth.get_distance(x0, y0)))
        cv2text(canvas, 'x0', worldCoord[0][0] , 20)
        cv2text(canvas, 'y0', worldCoord[0][1] , 50)
        cv2text(canvas, 'z0', worldCoord[0][2] , 80)
    if x1 > 0 and x1 < 1280 and y1 > 0 and y1 < 720:
        worldCoord.append(rs.rs2_deproject_pixel_to_point(intrinsics, [x1, y1], depth.get_distance(x1, y1)))
        cv2text(canvas, 'x1', worldCoord[1][0] , 110)
        cv2text(canvas, 'y1', worldCoord[1][1] , 140)
        cv2text(canvas, 'z1', worldCoord[1][2] , 170)
    return worldCoord


def eye_data_manage(left_img, right_img, left_points, right_points):
    left_img_ = cv2.resize(left_img,(50,50))
    left_points_ = left_points *  [50/left_img.shape[1], 50/left_img.shape[0], 1]
    left_img_ = np.mean(left_img_, axis = 2).astype('uint8')
    right_img_ = cv2.resize(right_img,(50,50))
    right_points_ = right_points * [50/right_img.shape[1], 50/right_img.shape[0], 1]
    right_img_ = np.mean(right_img_, axis = 2).astype('uint8')
    eye_img = np.concatenate([left_img_, right_img_], axis = 1)
    return eye_img, left_points_, right_points_


def eye_data_visual(eye_img, left_points_, right_points_):
    eye_canvas = eye_img.copy()
    right_points__ = right_points_.copy()
    right_points__[:, :1] = right_points__[:, :1] + 50
    cv2.circle(eye_canvas, left_points_[:,:2][0].astype('int'), 1, (255, 255, 255), 1)
    cv2.circle(eye_canvas, right_points__[:,:2][0].astype('int'), 1, (255, 255, 255), 1)
    cv2.imshow('eye', eye_canvas)