import carla
import numpy as np
import random
import pygame
import cv2

class World():
    def __init__(self, width, height, sync=False):
        self.width = width
        self.height = height

        # sensor
        self.camera_cms =None
        self.camera_game=None

        self.cms_size = [480, 320]
        self.pygame_surface = None

        self.cms_l_surface = None
        self.cms_r_surface = None

        # 前视参数 
        self.game_location = carla.Location(x=0.6, y=-0.35, z=1.25)

        # 左CMS参数
        self.cms_l_location = carla.Location(x=0.6, y=-1.1, z=1.0)

        # 右CMS参数
        self.cms_r_location = carla.Location(x=0.6, y=1.1, z=1.0)


        # 连接客户端
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        # 设定地图
        # self.world = self.client.load_world('Town10HD')

        # 获取世界
        self.world = self.client.get_world()

        # 初始设定
        self.settings = self.world.get_settings()
        self.original_settings = self.world.get_settings()

        # 初始化/重置
        self.restart(sync)

    def restart(self, sync=False):
        self.destory()
        self.sync_set(sync)
        self.car_set()
        self.sensor_set()
        self.spect_set()
        return
    
    def sync_set(self, sync):
        # 同步设置
        if sync:
            if not self.settings.synchronous_mode:
                self.settings.synchronous_mode = True
                self.settings.fixed_delta_seconds = 0.03
            self.world.apply_settings(self.settings)
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
        return
    
    def car_set(self):
        # 获取蓝图
        self.blueprint_library = self.world.get_blueprint_library()
        # 创建车
        self.car_bp = self.blueprint_library.filter("model3")[0]
        # 放置车
        car_transform = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.try_spawn_actor(self.car_bp, car_transform)
        return

    def sensor_set(self):
        # 创建游戏 RGB相机
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        sensor_transform = carla.Transform(carla.Location(x=0.6, y=-0.35, z=1.25))
        self.camera_bp.set_attribute('image_size_x', str(self.width))
        self.camera_bp.set_attribute('image_size_y', str(self.height))
        self.camera_bp.set_attribute('fov',str(110))
        self.camera_game = self.world.spawn_actor(self.camera_bp, sensor_transform, attach_to=self.car)
        # 监听RGB传感器
        self.camera_game.listen(lambda image: self.process_img(image, 'game'))

        # 创建CMS RGB相机
        self.camera_bp.set_attribute('image_size_x', str(self.cms_size[0]))
        self.camera_bp.set_attribute('image_size_y', str(self.cms_size[1]))
        self.camera_bp.set_attribute('fov',str(90))

        # 左CMS相机
        sensor_transform = carla.Transform(self.cms_l_location, carla.Rotation(yaw=180))
        self.camera_cms_l = self.world.spawn_actor(self.camera_bp, sensor_transform, attach_to=self.car)
        # 监听左CMS
        self.camera_cms_l.listen(lambda image: self.process_img(image, 'cms_l'))

        # 右CMS相机
        sensor_transform = carla.Transform(self.cms_r_location, carla.Rotation(yaw=180))
        self.camera_cms_r = self.world.spawn_actor(self.camera_bp, sensor_transform, attach_to=self.car)
        # 监听左CMS
        self.camera_cms_r.listen(lambda image: self.process_img(image, 'cms_r'))
        return

    # 控制cms的视角姿态
    def cms_control(self, param, rot=[0,180]):
        if param == 'l':
            new_transform = carla.Transform(self.cms_l_location, carla.Rotation(rot[0], rot[1], 0))
            self.camera_cms_l.set_transform(new_transform)   
        elif param == 'r':
            new_transform = carla.Transform(self.cms_r_location, carla.Rotation(rot[0], rot[1], 0))
            self.camera_cms_l.set_transform(new_transform) 
        elif param == 'game':
            new_transform = carla.Transform(self.game_location, carla.Rotation(rot[0], rot[1], 0))
            self.camera_game.set_transform(new_transform) 

        return

    # 设定观察者
    def spect_set(self):
        # 观察者视角
        spect_location = carla.Transform(self.car.get_transform().location + carla.Location(0, 0, 30), carla.Rotation(pitch=-90)) # 观察者位置
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(spect_location)
        return

    def process_img(self, image, param):
        # 渲染游戏窗口
        if param == 'game':
            array = np.array(image.raw_data)
            array = array.reshape(self.height, self.width, 4)
            array = array[:, :, :3]   
            rgb = array[:, :, ::-1]
            self.pygame_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

        # 渲染右CMS窗口
        elif param == 'cms_l':
            array = np.array(image.raw_data)
            array = array.reshape(self.cms_size[1], self.cms_size[0], 4)
            bgr = array[:, :, :3]   
            self.cms_l_surface = cv2.flip(bgr, 1)

        elif param == 'cms_r':
            array = np.array(image.raw_data)
            array = array.reshape(self.cms_size[1], self.cms_size[0], 4)
            bgr = array[:, :, :3]   
            self.cms_r_surface = cv2.flip(bgr, 1)
        
        return
    
    def destory(self):
        actors_lst = self.world.get_actors()
        for i in actors_lst:
            if i.type_id.split('.')[0] == 'sensor':
                i.stop()
                i.destroy()
                print('destroy a sensor')
            elif i.type_id.split('.')[0] == 'vehicle':
                i.destroy()
                print('destroy a vehicle')
        return
