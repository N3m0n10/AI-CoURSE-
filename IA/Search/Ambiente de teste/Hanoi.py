import pygame 
import queue
from random import randint

pygame.init()
sc = pygame.display.set_mode((800,600))
pygame.display.set_caption("Towers of Hanoi")
clock = pygame.time.Clock()

def move(x,y):
    y.put(x.get())

def elements(S):
    return list(S.queue)
    
S1 = queue.LifoQueue()
S2 = queue.LifoQueue()
S3 = queue.LifoQueue()
stacks = [S1, S2, S3]

for i in range(5):
    S1.put(i)

towers = len(stacks)
centers = [pygame.display.get_surface().get_width() // (towers + 1) * (i + 1) for i in range(towers)]
print(centers)
top_line = 340
line_height = 150
first_select = None
second_select = None

PIECES = 5
top_piece_y = top_line + line_height - 20*(PIECES + 2)  # actually opposite to top
rects = []
colors = [(randint(0,255),randint(0,255),randint(0,255)) for i in range(PIECES)]
def draw_rects(center):
    for piece in range(PIECES+1,1,-1):
        rects.append(pygame.Rect(center-piece*10, top_piece_y+piece*20, piece*20, 20))

draw_rects(centers[0])

## win condition: S3 == [1,2,3,4,5]
running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                pos = pygame.mouse.get_pos()
                for i, center in enumerate(centers):
                    if center - 10 < pos[0] < center + 10:
                        if first_select is None:
                            first_select = i
                        elif second_select is None and i != first_select:
                            second_select = i
                            if not stacks[first_select].empty():
                                rects[elements(stacks[first_select])[-1]].center = (centers[second_select], top_piece_y + line_height - ((len(elements(stacks[second_select])) + 1) * 20))  # what a messssssss
                                move(stacks[first_select], stacks[second_select]) 
                            first_select = None
                            second_select = None

    sc.fill((30,30,30))

    for i, s in enumerate(stacks): # force draw first
        pygame.draw.rect(sc, (120,150,200), (centers[i]-10, top_line, 20, line_height))

    for i, s in enumerate(stacks):
        for el in elements(s):
            rect = rects[el-1]
            #rect.center = (centers[i], 500 - (el * 20))
            pygame.draw.rect(sc, colors[el-1], rect)

    if elements(S3) == [0,1,2,3,4]:
        font = pygame.font.Font(None, 74)
        text = font.render("You Win!", True, (255, 255, 0))
        sc.blit(text, (300, 250))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()