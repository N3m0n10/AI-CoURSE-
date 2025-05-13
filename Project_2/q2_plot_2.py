import numpy as np
from Project_2.q2_value_iter import *
from matplotlib import pyplot as plt

def plot_q_table(q_table, grid_shape=(4, 12)):
    """Visualize Q-table with Q-values and policy arrows in each cell"""
    # Create matrix for text display
    text_matrix = np.empty(grid_shape, dtype=object)
    
    # Action to arrow mapping
    action_arrows = {
        'up': '↑',
        'down': '↓',
        'left': '←',
        'right': '→'
    }
    
    # Fill matrices
    for state_id, q_values in q_table.items():
        row = (state_id - 1) // grid_shape[1]
        col = (state_id - 1) % grid_shape[1]
        
        # Get best action
        best_action_idx = np.argmax(q_values)
        best_action = action_arrows[generic_actions[best_action_idx]]
        
        # Format Q-values with arrows
        q_text = [
            f"↑{q_values[2]:.2f}",
            f"↓{q_values[3]:.2f}",
            f"←{q_values[0]:.2f}",
            f"→{q_values[1]:.2f}"
        ]
        
        # Combine into multi-line text
        cell_text = "\n".join(q_text)
        text_matrix[row, col] = (cell_text, best_action)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Create grid
    ax.set_xlim(-0.5, grid_shape[1]-0.5)
    ax.set_ylim(-0.5, grid_shape[0]-0.5)
    
    # Draw grid lines
    for x in range(grid_shape[1]+1):
        ax.axvline(x-0.5, color='black')
    for y in range(grid_shape[0]+1):
        ax.axhline(y-0.5, color='black')
    
    # Add cell contents
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if (i == CLIFF_ROW and 0 < j < grid_shape[1]-1):
                # Cliff cells
                ax.text(j, i, 'X\n(Cliff)', ha='center', va='center', 
                        color='red', fontsize=8, weight='bold')
            elif j == grid_shape[1]-1 and i == CLIFF_ROW:
                # Goal cell
                ax.text(j, i, 'GOAL', ha='center', va='center', 
                       color='green', fontsize=12, weight='bold')
            else:
                # Regular cells
                cell_text, best_action = text_matrix[i, j]
                ax.text(j, i, 
                       f"{best_action}\n––––––\n{cell_text}", 
                       ha='center', va='center', 
                       fontsize=6, linespacing=1.5,
                       bbox=dict(facecolor='white', alpha=0.8))
    
    # Set ticks and labels
    ax.set_xticks(np.arange(grid_shape[1]))
    ax.set_yticks(np.arange(grid_shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.invert_yaxis()  # To match matrix coordinates
    ax.set_title('Q-Table Visualization (Values and Optimal Policy)', fontsize=14)
    
    plt.show()



# Use the function
trained = q_learning()
plot_q_table(trained)