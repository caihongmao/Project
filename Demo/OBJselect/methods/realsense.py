# realsense设置
import pyrealsense2 as rs
import numpy as np

# def color_depth(frames):
#     depth_frame = frames.get_depth_frame()
#     color_frame = frames.get_color_frame()
#     color_image = np.asanyarray(color_frame.get_data())
#     return color_image, depth_frame
            
# def realsenseConfig():
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#     config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#     # 对齐设置
#     align_to = rs.stream.color
#     alignedFs = rs.align(align_to)
#     pipeline.start(config)
#     return pipeline, alignedFs

# 获取世界坐标
def getworldCoord(x, y, depth_frame, intrinsics):
    depth_value = depth_frame.get_distance(x, y)
    camera_point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_value)
    return camera_point

def color_depth(frames):
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    ir_frame = frames.get_infrared_frame()
    color_image = np.asanyarray(color_frame.get_data())
    ir_image = np.asanyarray(ir_frame.get_data())
    return color_image, depth_frame, ir_image
            
def realsenseConfig():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)

    # 对齐设置
    align_to = rs.stream.color
    alignedFs = rs.align(align_to)
    pipeline.start(config)
    return pipeline, alignedFs
