import pygame 

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