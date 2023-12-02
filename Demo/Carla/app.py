import carla
from carla import ColorConverter as cc

import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_0
from pygame.locals import K_9
from pygame.locals import K_BACKQUOTE
from pygame.locals import K_BACKSPACE
from pygame.locals import K_COMMA
from pygame.locals import K_DOWN
from pygame.locals import K_ESCAPE
from pygame.locals import K_F1
from pygame.locals import K_LEFT
from pygame.locals import K_PERIOD
from pygame.locals import K_RIGHT
from pygame.locals import K_SLASH
from pygame.locals import K_SPACE
from pygame.locals import K_TAB
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_c
from pygame.locals import K_d
from pygame.locals import K_h
from pygame.locals import K_m
from pygame.locals import K_p
from pygame.locals import K_q
from pygame.locals import K_r
from pygame.locals import K_s
from pygame.locals import K_w

import numpy as np
import cv2
import random
import math
from configparser import ConfigParser

# # 创建全屏窗口
# cv2.namedWindow('CMS', cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty('CMS', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

class World():
    def __init__(self, width, height, sync=False):
        self.width = width
        self.height = height
        self.pygame_surface = None
        self.cms_size = [480, 320]
        self.cms_surface = None

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
        self.camera_bp.set_attribute('fov',str(130))
        self.camera_game = self.world.spawn_actor(self.camera_bp, sensor_transform, attach_to=self.car)
        # 监听RGB传感器
        self.camera_game.listen(lambda image: self.process_img(image, 'game'))

        # 创建CMS RGB相机
        sensor_transform = carla.Transform(carla.Location(x=0.6, y=-1.1, z=1.0), carla.Rotation(yaw=0))
        self.camera_bp.set_attribute('image_size_x', str(self.cms_size[0]))
        self.camera_bp.set_attribute('image_size_y', str(self.cms_size[1]))
        self.camera_bp.set_attribute('fov',str(90))
        self.camera_cms = self.world.spawn_actor(self.camera_bp, sensor_transform, attach_to=self.car)

        
        # 监听RGB传感器
        self.camera_cms.listen(lambda image: self.process_img(image, 'cms'))
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

        # 渲染CMS窗口
        elif param == 'cms':
            array = np.array(image.raw_data)
            array = array.reshape(self.cms_size[1], self.cms_size[0], 4)
            bgr = array[:, :, :3]   
            self.cms_surface = cv2.flip(bgr, 1)
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

class Controller():
    def __init__(self, joystick=False):

        # 阿克曼模拟
        self._ackermann_enabled = False
        # self._ackermann_control = carla.VehicleAckermannControl()
        self._ackermann_reverse = 1

        # 普通模拟
        self._control = carla.VehicleControl()

        self.throttle = 1
        self.joystick = joystick

        # G29初始化
        if self.joystick:
            pygame.joystick.init()
            joystick_count = pygame.joystick.get_count()
            if joystick_count > 1:
                raise ValueError("Please Connect Just One Joystick")
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            self._parser = ConfigParser()
            self._parser.read('wheel_config.ini')

            # 按键定义
            self._steer_idx = int(
                self._parser.get('G29 Racing Wheel', 'steering_wheel'))
            self._throttle_idx = int(
                self._parser.get('G29 Racing Wheel', 'throttle'))
            self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
            self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
            self._handbrake_idx = int(self._parser.get('G29 Racing Wheel', 'handbrake'))
    
    def parse_event(self, world, clock):
        # 设置按键
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                # elif event.button == 1:
                #     world.hud.toggle_info()
                # elif event.button == 2:
                #     world.camera_manager.toggle_camera()
                # elif event.button == 3:
                #     world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                # elif event.button == 23:
                #     world.camera_manager.next_sensor()
                    
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
        # 车控按键
        self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
        if self.joystick:
            self._parse_vehicle_wheel()
        self._control.reverse = self._control.gear < 0
        
        # 实施车控
        if not self._ackermann_enabled:
            world.car.apply_control(self._control)
        else:
            world.car.apply_ackermann_control(self._ackermann_control)
            self._control = world.car.get_control()

    def _is_quit_shortcut(self, key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
    
    # 按键功能
    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.1, 1.00)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)
    
    # G29功能
    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

class HUD():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.speed = 0

    def tick(self, world):
        v = world.car.get_velocity()
        self.speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    
    # 绘制圆形仪表
    def draw_dashboard(self, display, value, label, x_shift, y_shift, range):
        # 定义颜色
        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (255, 0, 0)
        ph = self.height / 720
        pw = self.width / 1280

        center_x = self.width // 2 + int(x_shift * pw)
        center_y = self.height // 2 + int(y_shift * ph)

         # 绘制背景颜色
        pygame.draw.circle(display, black, (center_x, center_y), 100)

        # 绘表盘外圈
        pygame.draw.circle(display, white, (center_x, center_y), 100, 2)

        # 绘制表盘
        font1 = pygame.font.Font(None, 64)
        font2 = pygame.font.Font(None, 24)

        if int(value) < 0:
            value = 0

        text = font1.render(str(int(value)), True, white)
        label_text = font2.render(label, True, white)
        
        # 字符长度 居中
        if len(str(int(value))) == 1:
            shift = 10
        elif len(str(int(value))) == 2:
            shift = 25
        elif len(str(int(value))) == 3:
            shift = 40
        else :
            shift = 55

        display.blit(text, ((center_x - shift, center_y - 80)))
        display.blit(label_text, ((center_x - 16, center_y - 40)))

        # 绘制指示器
        indicator_length = 80
        indicator_angle = 180 + value / range * 360
        indicator_x = center_x + int(indicator_length * math.sin(math.radians(indicator_angle)))
        indicator_y = center_y - int(indicator_length * math.cos(math.radians(indicator_angle)))
        pygame.draw.line(display, red, (center_x, center_y), (indicator_x, indicator_y), 4)

class Game():
    def __init__(self, width, height):
        # window初始化
        pygame.init()
        pygame.font.init()

        # 渲染设置
        self.display = pygame.display.set_mode(
            (width, height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.display.fill((0,0,0))
        pygame.display.flip()

    def render(self, display, surface):
        if surface is not None:
            display.blit(surface, (0, 0))

class Runner():
    def __init__(self, width=1280, height=720, joystick=True):
        self.game = Game(width, height)
        self.env = World(width, height, False)
        self.hud = HUD(width, height)
        self.controller = Controller(joystick)

    def run(self):
        clock = pygame.time.Clock()
        try:
            while True:
                clock.tick_busy_loop(60)
                if self.env.settings.synchronous_mode:
                    self.env.world.tick()

                # 观察者跟随
                # self.env.spect_set()
                
                if self.controller.parse_event(self.env, clock):
                    return
                self.hud.tick(self.env)

                # 绘制速度表
                self.hud.draw_dashboard(self.game.display, self.hud.speed, 'km/h',-100, 240, 180)
                # 绘制转速表
                self.hud.draw_dashboard(self.game.display, (self.controller._control.throttle - self.controller._control.brake) * 4000, 'rpm', 100, 240, 4000)

                pygame.display.flip()
                self.game.render(self.game.display, self.env.pygame_surface)

                if self.env.cms_surface is not None:
                    cv2.imshow('window', self.env.cms_surface)
                    cv2.waitKey(2)
                
        finally:
            self.env.destory()
            self.env.world.apply_settings(self.env.original_settings)
            pygame.quit()
    
runner = Runner(width=1280, height=720, joystick=False)
runner.run()