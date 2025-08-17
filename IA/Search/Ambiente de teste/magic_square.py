"""GEMINI assisted"""
import pygame
import sys
import random

pygame.init()
sc = pygame.display.set_mode((600, 600), pygame.RESIZABLE)
pygame.display.set_caption("Magic Square Puzzle")
# The game_surface is a fixed-resolution surface where we draw our game elements.
game_surface = pygame.Surface((600, 600))

SIZE = 3
if SIZE > 10: SIZE = 10 # Limit size to keep text readable

# Magic constant formula: n * (n^2 + 1) / 2
TARGET = SIZE * (SIZE**2 + 1) / 2

# Create a list of numbers from 1 to SIZE^2 and shuffle them
nums = list(range(1, SIZE**2 + 1))
random.shuffle(nums)

# Populate the SQUARE grid with the shuffled numbers
SQUARE = []
for i in range(SIZE):
    row = nums[i * SIZE : (i + 1) * SIZE]
    SQUARE.append(row)

# --- State and Style Variables ---
POS = (0, 0) # The (row, col) of the active cursor
selected_pos = None # Stores the first cell selected for a swap
is_solved = False

# Fonts
FONT = pygame.font.SysFont("sans-serif", 50)
SUM_FONT = pygame.font.SysFont("sans-serif", 25)
WIN_FONT = pygame.font.SysFont("sans-serif", 100)

# Colors
COLOR_BG = (30, 30, 30)
COLOR_GRID = (100, 100, 100)
COLOR_TEXT = (240, 240, 240)
COLOR_CURSOR = (255, 215, 0)  # Gold, for the active cursor
COLOR_SELECT = (0, 191, 255)   # Deep Sky Blue, for the first selection
COLOR_SUM = (255, 80, 80)      # Red, for sums
COLOR_WIN = (60, 220, 120)     # Green

# Layout constants
MARGIN = 60
GRID_AREA_SIZE = 600 - MARGIN


def diagonal(square):
    return [square[i][i] for i in range(SIZE)]

def anti_diagonal(square):
    return [square[i][SIZE - 1 - i] for i in range(SIZE)]

def collums(square):
    return [[square[j][i] for j in range(SIZE)] for i in range(SIZE)]

def lines(square):
    return square

def check_magic_square(square):
    col, rol = collums(square), lines(square)

    # Check all columns
    for i in range(SIZE):
        if sum(col[i]) != TARGET:
            return False
            
    # Check all rows
    for i in range(SIZE):
        if sum(rol[i]) != TARGET:
            return False
            
    # Check main diagonal
    if sum(diagonal(square)) != TARGET:
        return False

    # Check anti-diagonal
    if sum(anti_diagonal(square)) != TARGET:
        return False
        
    return True

def swap(square, pos1, pos2):
    y1, x1 = pos1 # Unpack as row, col
    y2, x2 = pos2
    square[y1][x1], square[y2][x2] = square[y2][x2], square[y1][x1]

# --- Main game loop ---
running = True
while running:
    # --- Handle Events ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.VIDEORESIZE:
            sc = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        # Handle keyboard input for movement and selection
        if event.type == pygame.KEYDOWN and not is_solved:
            r, c = POS
            if event.key == pygame.K_RIGHT:
                POS = (r, (c + 1) % SIZE)
            elif event.key == pygame.K_LEFT:
                POS = (r, (c - 1 + SIZE) % SIZE)
            elif event.key == pygame.K_DOWN:
                POS = ((r + 1) % SIZE, c)
            elif event.key == pygame.K_UP:
                POS = ((r - 1 + SIZE) % SIZE, c)
            
            # Handle ENTER key for selection and swapping
            elif event.key == pygame.K_RETURN:
                if selected_pos is None:
                    # First selection
                    selected_pos = POS
                elif selected_pos == POS:
                    # Deselect if ENTER is pressed on the same square
                    selected_pos = None
                else:
                    # Second selection: perform the swap
                    swap(SQUARE, selected_pos, POS)
                    selected_pos = None # Reset selection after swap
                    
                    if check_magic_square(SQUARE):
                        is_solved = True
                        print("Magic square found!")


    # --- Drawing Logic ---
    game_surface.fill(COLOR_BG)
    cell_size = GRID_AREA_SIZE // SIZE

    # Draw the grid and numbers
    for r in range(SIZE):
        for c in range(SIZE):
            rect = pygame.Rect(
                c * cell_size + MARGIN // 2, 
                r * cell_size + MARGIN // 2, 
                cell_size, cell_size
            )
            
            # Draw grid cell border
            pygame.draw.rect(game_surface, COLOR_GRID, rect, 1)

            # Draw highlight for the first selected number
            if selected_pos == (r, c):
                 pygame.draw.rect(game_surface, COLOR_SELECT, rect, 4)
            
            # Draw the mask for the current cursor position
            if POS == (r, c):
                 pygame.draw.rect(game_surface, COLOR_CURSOR, rect, 4)

            # Render and draw the number
            num_text = FONT.render(str(SQUARE[r][c]), True, COLOR_TEXT)
            text_rect = num_text.get_rect(center=rect.center)
            game_surface.blit(num_text, text_rect)

    # Calculate and display ALL sums near the border in red
    # Column Sums (Top)
    for c in range(SIZE):
        col_sum = sum(SQUARE[r][c] for r in range(SIZE))
        sum_text = SUM_FONT.render(str(col_sum), True, COLOR_SUM)
        pos_x = c * cell_size + cell_size // 2 + MARGIN // 2
        text_rect = sum_text.get_rect(center=(pos_x, MARGIN // 4))
        game_surface.blit(sum_text, text_rect)

    # Row Sums (Right)
    for r in range(SIZE):
        row_sum = sum(SQUARE[r])
        sum_text = SUM_FONT.render(str(row_sum), True, COLOR_SUM)
        pos_y = r * cell_size + cell_size // 2 + MARGIN // 2
        text_rect = sum_text.get_rect(center=(600 - MARGIN // 4, pos_y))
        game_surface.blit(sum_text, text_rect)

    # Diagonal Sums (Corners)
    diag1_sum = sum(diagonal(SQUARE))
    diag2_sum = sum(anti_diagonal(SQUARE))
    d1_text = SUM_FONT.render(str(diag1_sum), True, COLOR_SUM)
    d2_text = SUM_FONT.render(str(diag2_sum), True, COLOR_SUM)
    game_surface.blit(d1_text, d1_text.get_rect(center=(600 - MARGIN // 4, 600 - MARGIN // 4)))
    game_surface.blit(d2_text, d2_text.get_rect(center=(600 - MARGIN // 4, MARGIN // 4)))

    # If the square is solved, display a "You Win!" message
    if is_solved:
        win_surf = WIN_FONT.render("You Win!", True, COLOR_WIN)
        win_rect = win_surf.get_rect(center=(300, 300))
        # Add a semi-transparent overlay for better readability
        overlay = pygame.Surface((600, 600), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        game_surface.blit(overlay, (0,0))
        game_surface.blit(win_surf, win_rect)

    # blit surface to screen (resized)
    scaled_surface = pygame.transform.scale(game_surface, sc.get_size())
    sc.blit(scaled_surface, (0, 0))

    pygame.display.flip()

pygame.quit()
sys.exit()