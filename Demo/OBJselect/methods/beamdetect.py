import numpy as np
import cv2
  
def beam_detect(ray_origin, ray_direction, target_list, param):
    detect_lst = []

    # 执行球体检测
    if param == 'sphere':
        for key, value in target_list.items():
            res = sphere_detect(ray_origin, ray_direction, value[:3], value[-1])
            if res is not False:
                detect_lst.append(key)

    # 执行矩形检测
    if param == 'rectangle':
         for key, value in target_list.items():
            res = rectangle_detect(ray_origin, ray_direction, value)
            if res is not False:
                detect_lst.append(key)

    return detect_lst
    

# 球体检测
def sphere_detect(ray_origin, ray_direction, sphere_center, sphere_radius):
    # Calculate the vector from the ray origin to the sphere center
    sphere_to_ray = ray_origin - sphere_center
    
    # Calculate the quadratic equation coefficients
    a = np.dot(ray_direction, ray_direction)
    b = 2 * np.dot(ray_direction, sphere_to_ray)
    c = np.dot(sphere_to_ray, sphere_to_ray) - sphere_radius ** 2
    
    # Calculate the discriminant
    discriminant = b ** 2 - 4 * a * c
    
    if discriminant < 0:
        # No intersection
        return False
    else:
        # Calculate the two possible solutions for t (parameter along the ray)
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        if t1 >= 0 or t2 >= 0:
            # Ray intersects the sphere
            intersection_point = ray_origin + min(t1, t2) * ray_direction
            return intersection_point
        else:
            # Ray originates inside the sphere, no intersection
            return False

#  矩形检测     
def rectangle_detect(ray_origin, ray_direction, rectangle_vertices):
    # 将输入转换为NumPy数组以便进行矩阵运算
    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction)
    rectangle_vertices = np.array(rectangle_vertices)

    # 计算矩形的两个边向量
    edge1 = rectangle_vertices[1] - rectangle_vertices[0]
    edge2 = rectangle_vertices[3] - rectangle_vertices[0]

    # 计算法线向量
    normal = np.cross(edge1, edge2)

    # 计算射线与矩形平面的交点
    t = np.dot(rectangle_vertices[0] - ray_origin, normal) / np.dot(ray_direction, normal)

    if t >= 0:
        # 计算交点的坐标
        intersection_point = ray_origin + t * ray_direction

        # 检查交点是否在矩形内部
        min_x = min(rectangle_vertices[:, 0])
        max_x = max(rectangle_vertices[:, 0])
        min_y = min(rectangle_vertices[:, 1])
        max_y = max(rectangle_vertices[:, 1])
        min_z = min(rectangle_vertices[:, 2])
        max_z = max(rectangle_vertices[:, 2])

        if min_x <= intersection_point[0] <= max_x and \
           min_y <= intersection_point[1] <= max_y and \
           min_z <= intersection_point[2] <= max_z:
            return True

    return False


def beam_obj(coord, gaze_vector, sphere_list, rectangle_list):
    obj = []
    res_1 = beam_detect(coord, gaze_vector, sphere_list, 'sphere')
    res_2 = beam_detect(coord, gaze_vector, rectangle_list, 'rectangle')
    if len(res_1) > 0:
        obj.append(res_1)
    if len(res_2) > 0:
        obj.append(res_2) 
    obj_ = np.array(obj).flatten()
    img, select_state = obj_visualize(obj_)
    return img, select_state
    

def obj_visualize(obj_):
    
    #可视化
    width, height = 640, 480
    select_state = [0,0,0,0,0,0]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if len(obj_) > 0 and 'a' in obj_:
        cv2.rectangle(image, (460, 80), (560, 160), (0, 255, 0), 100)
        select_state[0] = 1
    else:    
        cv2.rectangle(image, (460, 80), (560, 160), (0, 0, 255), 100)

    if len(obj_) > 0 and 'b' in obj_:
        cv2.rectangle(image, (80, 80), (180, 160), (0, 255, 0), 100)
        select_state[1] = 1
    else:    
        cv2.rectangle(image, (80, 80), (180, 160), (0,  0, 255), 100)

    if len(obj_) > 0 and 'd' in obj_:
        cv2.rectangle(image, (80, 280), (180, 360), (0, 255, 0), 100)
        select_state[2] = 1
    else:    
        cv2.rectangle(image, (80, 280), (180, 360), (0, 0, 255), 100)

    if len(obj_) > 0 and 'c' in obj_:
        cv2.rectangle(image, (460, 280), (560, 360), (0, 255, 0), 100)
        select_state[3] = 1
    else:    
        cv2.rectangle(image, (460, 280), (560, 360), (0, 0, 255), 100)

    if len(obj_) > 0 and 'e' in obj_:
        cv2.rectangle(image, (300, 260), (340, 300), (0, 255, 0), 100)
        select_state[4] = 1
    else:    
        cv2.rectangle(image, (300, 260), (340, 300), (0, 0, 255), 100)

    if len(obj_) > 0 and 'f' in obj_:
        cv2.rectangle(image, (300, 100), (340, 140), (0, 255, 0), 100)
        select_state[5] = 1
    else:    
        cv2.rectangle(image, (300, 100), (340, 140), (0, 0, 255), 100)

    return image, select_state

def tranfsorm_func(obj, th):
    def transform(x, y, a):
        theta = a / 180 * np.pi
        xn = x * np.cos(theta) - y * np.sin(theta)
        yn = x * np.sin(theta) + y * np.cos(theta)
        return xn, yn
    for key, value  in obj.items():
        if type(value[0]) == list:
            for i in value:
                z_, y_ = transform(-i[2], i[1], th)
                i[2] = -z_
                i[1] = y_
        else:
            z_, y_ = transform(-value[2], value[1], th)
            value[2] = -z_
            value[1] = y_
