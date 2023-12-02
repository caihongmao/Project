import carla
from carla import ColorConverter as cc

import pygame
import numpy as np
import cv2
import random
from func.control import *
from func.hud import *
from func.world import *
from func.game import *
import time

pitch = 0
yaw = 0

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
                self.hud.draw_dashboard(self.game.display, self.hud.speed, 'km/h',-120, 240, 180)
                # 绘制转速表
                self.hud.draw_dashboard(self.game.display, (self.controller._control.throttle - self.controller._control.brake) * 4000, 'rpm', 100, 240, 4000)

                pygame.display.flip()
                self.game.render(self.game.display, self.env.pygame_surface)

                # 控制视角
                self.env.cms_control('l', [-pitch, yaw])
                
                if self.env.cms_l_surface is not None:
                    cv2.imshow('cms_l', self.env.cms_l_surface)
                    cv2.waitKey(2)
                
                if self.env.cms_r_surface is not None:
                    cv2.imshow('cms_r', self.env.cms_r_surface)
                    cv2.waitKey(2)
                
        finally:
            self.env.destory()
            self.env.world.apply_settings(self.env.original_settings)
            pygame.quit()



import socketio

sio = socketio.Client()

@sio.on('face_keypoints')
def handle_face_keypoints(data):
    global pitch, yaw
    print('Received face keypoints:', data)
    pitch = data['gaze'][0]['head_rot'][0]
    yaw = data['gaze'][0]['head_rot'][1] - 180


sio.connect('http://127.0.0.1:5000')
runner = Runner(width=1280, height=720, joystick=False)  
runner.run()

   
