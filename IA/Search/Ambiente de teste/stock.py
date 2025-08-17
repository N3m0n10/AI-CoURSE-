import pygame
from copy import deepcopy

WIDTH, HEIGHT = 800, 600
pygame.init()
sc = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Stock")
clock = pygame.time.Clock()

bays = [['a','b','c'],['d','e'],['f','g']]
TARGET = [ [] , ['a','b','c','g','f'] , ['e','d'] ]

font = pygame.font.Font(None, 50)
win_font = pygame.font.Font(None, 250)
win_text = win_font.render("You win!", True, (0, 220, 0))
target_text = font.render("Target: " + str(TARGET), True, (255, 255, 255))

bays_rects = [pygame.Rect(130 + i * 200, 100, 150, 350) for i in range(len(bays))]
selected_1 = None

def move(from_bay, to_bay):
    if len(from_bay) > 0:
        to_bay.append(from_bay.pop())

def draw_bays():
    for i, rect in enumerate(bays_rects):
        pygame.draw.rect(sc, (240, 240, 240), rect)
        for j, item in enumerate(bays[i]):
            text = font.render(item, True, (0, 0, 0))
            sc.blit(text, (rect.centerx, rect.y + 300 - j * 50))

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                pos = pygame.mouse.get_pos()
                for i, rect in enumerate(bays_rects):
                    if rect.collidepoint(pos):
                        if len(bays[i]) > 0 and selected_1 is None:
                            selected_1 = i 
                            print(f"Selected bay: {i}")
                        elif selected_1 is not None and i == selected_1:
                            selected_1 = None
                            print("Deselected bay")
                        elif selected_1 is not None and i != selected_1:
                            move(bays[selected_1], bays[i]) 
                            selected_1 = None
                            print(f"Moved from bay {selected_1} to bay {i}")
        if event.type == pygame.KEYDOWN: # cheat
            if event.key == pygame.K_SPACE:
                bays = deepcopy(TARGET)

    sc.fill((30, 30, 30))
    sc.blit(target_text, (WIDTH//2 - target_text.get_width()//2, 10))
    draw_bays()

    if all(bays[i] == TARGET[i] for i in range(len(bays)) ):
        sc.blit(win_text, (WIDTH // 2 - win_text.get_width() // 2, HEIGHT // 2 - win_text.get_height() // 2))

    pygame.display.flip()

pygame.quit()

    