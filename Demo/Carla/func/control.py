import pygame
import carla
from configparser import ConfigParser
import math

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
