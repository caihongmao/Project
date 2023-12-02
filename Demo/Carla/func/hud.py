import math 
import pygame

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
