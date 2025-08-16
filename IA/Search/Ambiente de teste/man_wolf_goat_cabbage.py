import pygame
import sys

# Initialize Pygame
pygame.init()

# --- Constants and Configuration ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (135, 206, 250)  # Sky Blue
WATER_COLOR = (64, 164, 223)
BANK_COLOR = (34, 139, 34)  # Forest Green
FONT_COLOR = (255, 255, 255)
GAMEOVER_COLOR = (200, 0, 0)
WIN_COLOR = (0, 150, 0)

# Game states
GAME_ACTIVE = 0
GAME_OVER = 1
GAME_WIN = 2

# --- Setup the Display ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Man, Wolf, Goat, and Cabbage Puzzle")
font = pygame.font.Font(None, 48)
small_font = pygame.font.Font(None, 24)

# --- Game Object Representation ---
# We will use simple rectangles to represent the game objects for simplicity.
# You could replace these with loaded images.
SPRITE_SIZE = 50
sprites = {
    'man': {'color': (255, 222, 173), 'label': 'Man'},
    'wolf': {'color': (128, 128, 128), 'label': 'Wolf'},
    'goat': {'color': (255, 255, 255), 'label': 'Goat'},
    'cabbage': {'color': (0, 255, 0), 'label': 'Cabbage'},
    'boat': {'color': (139, 69, 19), 'label': 'Boat'}
}

# --- Game State Variables ---
def reset_game():
    """Resets all game variables to their initial state."""
    global locations, boat_side, boat_contents, game_state, message
    
    # 'left' or 'right' bank for each sprite
    locations = {
        'man': 'left',
        'wolf': 'left',
        'goat': 'left',
        'cabbage': 'left'
    }
    boat_side = 'left'
    boat_contents = [] # Can contain one item besides the man
    game_state = GAME_ACTIVE
    message = ""

# Initialize game state
reset_game()

# --- Positions and Clickable Rects ---
# Define static positions for sprites on banks, in the boat, etc.
positions = {
    'left_bank': [(50, 250), (120, 250), (50, 320), (120, 320)],
    'right_bank': [(SCREEN_WIDTH - 100, 250), (SCREEN_WIDTH - 170, 250), (SCREEN_WIDTH - 100, 320), (SCREEN_WIDTH - 170, 320)],
    'boat_left': (250, 450),
    'boat_right': (SCREEN_WIDTH - 350, 450),
    'passenger_left': (310, 450),
    'passenger_right': (SCREEN_WIDTH - 290, 450)
}

# Create rects for clicking
sprite_rects = {}

# --- Helper Functions ---
def draw_background():
    """Draws the static background elements like river and banks."""
    screen.fill(BACKGROUND_COLOR)
    pygame.draw.rect(screen, WATER_COLOR, (0, 400, SCREEN_WIDTH, 200))
    pygame.draw.rect(screen, BANK_COLOR, (0, 200, 200, 200)) # Left bank
    pygame.draw.rect(screen, BANK_COLOR, (SCREEN_WIDTH - 200, 200, 200, 200)) # Right bank

def draw_sprites():
    """Draws all sprites based on their current locations."""
    global sprite_rects
    sprite_rects = {} # Reset rects each frame

    # --- Draw Sprites on Banks ---
    left_bank_sprites = [s for s, loc in locations.items() if loc == 'left']
    right_bank_sprites = [s for s, loc in locations.items() if loc == 'right']

    for i, sprite_name in enumerate(left_bank_sprites):
        pos = positions['left_bank'][i]
        color = sprites[sprite_name]['color']
        rect = pygame.Rect(pos[0], pos[1], SPRITE_SIZE, SPRITE_SIZE)
        pygame.draw.rect(screen, color, rect)
        sprite_rects[sprite_name] = rect
        # Draw label
        label_surf = small_font.render(sprites[sprite_name]['label'], True, (0,0,0))
        screen.blit(label_surf, (pos[0] + 5, pos[1] + SPRITE_SIZE + 5))


    for i, sprite_name in enumerate(right_bank_sprites):
        pos = positions['right_bank'][i]
        color = sprites[sprite_name]['color']
        rect = pygame.Rect(pos[0], pos[1], SPRITE_SIZE, SPRITE_SIZE)
        pygame.draw.rect(screen, color, rect)
        sprite_rects[sprite_name] = rect
        # Draw label
        label_surf = small_font.render(sprites[sprite_name]['label'], True, (0,0,0))
        screen.blit(label_surf, (pos[0] + 5, pos[1] + SPRITE_SIZE + 5))

    # --- Draw Boat and its occupants ---
    boat_pos = positions['boat_left'] if boat_side == 'left' else positions['boat_right']
    boat_rect = pygame.Rect(boat_pos[0], boat_pos[1], 120, 50)
    pygame.draw.rect(screen, sprites['boat']['color'], boat_rect)
    sprite_rects['boat'] = boat_rect # Make the boat itself clickable to move

    # Man is always in the boat if it's not on his bank
    if locations['man'] == 'boat':
        man_pos = boat_pos
        man_rect = pygame.Rect(man_pos[0] + 5, man_pos[1], SPRITE_SIZE, SPRITE_SIZE)
        pygame.draw.rect(screen, sprites['man']['color'], man_rect)
        sprite_rects['man'] = man_rect

    # Draw passenger
    if boat_contents:
        passenger_name = boat_contents[0]
        passenger_pos = positions['passenger_left'] if boat_side == 'left' else positions['passenger_right']
        passenger_rect = pygame.Rect(passenger_pos[0], passenger_pos[1], SPRITE_SIZE, SPRITE_SIZE)
        pygame.draw.rect(screen, sprites[passenger_name]['color'], passenger_rect)
        sprite_rects[passenger_name] = passenger_rect


def check_game_over():
    """Checks for losing conditions."""
    global game_state, message

    # Check left bank
    left_bank_items = {s for s, loc in locations.items() if loc == 'left'}
    if locations['man'] != 'left':
        if 'wolf' in left_bank_items and 'goat' in left_bank_items:
            game_state = GAME_OVER
            message = "The wolf ate the goat!"
            return
        if 'goat' in left_bank_items and 'cabbage' in left_bank_items:
            game_state = GAME_OVER
            message = "The goat ate the cabbage!"
            return

    # Check right bank
    right_bank_items = {s for s, loc in locations.items() if loc == 'right'}
    if locations['man'] != 'right':
        if 'wolf' in right_bank_items and 'goat' in right_bank_items:
            game_state = GAME_OVER
            message = "The wolf ate the goat!"
            return
        if 'goat' in right_bank_items and 'cabbage' in right_bank_items:
            game_state = GAME_OVER
            message = "The goat ate the cabbage!"
            return

def check_win():
    """Checks for the winning condition."""
    global game_state, message
    if all(loc == 'right' for loc in locations.values()):
        game_state = GAME_WIN
        message = "Congratulations! You solved the puzzle!"


def handle_click(pos):
    """Processes a mouse click event."""
    global boat_side # Moved this declaration to the top of the function
    
    if game_state != GAME_ACTIVE:
        # If game is over or won, any click might restart it (e.g. on a button)
        if reset_button_rect.collidepoint(pos):
            reset_game()
        return

    # Check if a sprite was clicked
    for sprite_name, rect in sprite_rects.items():
        if rect.collidepoint(pos):
            # --- Logic for moving items to/from the boat ---
            if sprite_name != 'boat' and sprite_name != 'man':
                # If item is on the same bank as the boat and the man
                if locations[sprite_name] == boat_side and locations['man'] == boat_side:
                    if not boat_contents: # If boat is empty
                        locations[sprite_name] = 'boat'
                        boat_contents.append(sprite_name)
                        return
                # If item is in the boat
                elif locations[sprite_name] == 'boat':
                    locations[sprite_name] = boat_side
                    boat_contents.remove(sprite_name)
                    return

            # --- Logic for moving the man and the boat ---
            if sprite_name == 'man' and locations['man'] == boat_side:
                # Move the man to the boat to initiate crossing
                locations['man'] = 'boat'
                return
            
            # If boat is clicked (or man in boat is clicked) and man is in boat
            if (sprite_name == 'boat' or sprite_name == 'man') and locations['man'] == 'boat':
                # Cross the river
                boat_side = 'right' if boat_side == 'left' else 'left'
                # Man arrives at the new bank
                locations['man'] = boat_side
                # Passenger also arrives
                if boat_contents:
                    locations[boat_contents[0]] = boat_side
                    boat_contents.clear()
                
                # After moving, check for game over or win
                check_game_over()
                if game_state == GAME_ACTIVE: # Only check for win if not game over
                    check_win()
                return


# --- Main Game Loop ---
running = True
while running:
    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_click(event.pos)

    # Drawing
    draw_background()
    draw_sprites()

    # --- Display Game State Messages ---
    if game_state == GAME_OVER:
        text_surf = font.render("GAME OVER", True, GAMEOVER_COLOR)
        text_rect = text_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 50))
        screen.blit(text_surf, text_rect)
        
        msg_surf = small_font.render(message, True, FONT_COLOR)
        msg_rect = msg_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        screen.blit(msg_surf, msg_rect)

    elif game_state == GAME_WIN:
        text_surf = font.render("YOU WIN!", True, WIN_COLOR)
        text_rect = text_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 50))
        screen.blit(text_surf, text_rect)

        msg_surf = small_font.render(message, True, FONT_COLOR)
        msg_rect = msg_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        screen.blit(msg_surf, msg_rect)
        
    # Draw Reset Button
    reset_button_rect = pygame.Rect(SCREEN_WIDTH / 2 - 75, SCREEN_HEIGHT - 70, 150, 50)
    pygame.draw.rect(screen, (200, 200, 200), reset_button_rect, border_radius=10)
    reset_text_surf = font.render("Reset", True, (0, 0, 0))
    reset_text_rect = reset_text_surf.get_rect(center=reset_button_rect.center)
    screen.blit(reset_text_surf, reset_text_rect)


    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()