import numpy as np
from tst_q2 import *
from matplotlib import pyplot as plt

def plot_q_table(q_table, grid_shape=(4, 12)):
    """Visualize Q-table as a heatmap with policy arrows"""
    # Create value matrix
    value_matrix = np.zeros(grid_shape)
    policy_matrix = np.empty(grid_shape, dtype=str)
    
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
        
        # Store maximum Q-value
        value_matrix[row, col] = max(q_values)
        
        # Store best action
        best_action_idx = np.argmax(q_values)
        policy_matrix[row, col] = action_arrows[generic_actions[best_action_idx]]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot value heatmap
    im1 = ax1.imshow(value_matrix)
    ax1.set_title('State Values')
    plt.colorbar(im1, ax=ax1)
    
    # Plot policy arrows
    im2 = ax2.imshow(value_matrix, cmap='RdYlBu')
    ax2.set_title('Optimal Policy')
    
    # Add arrows for policy
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if (i == CLIFF_ROW and 0 < j < grid_shape[1]-1):
                ax2.text(j, i, 'X', ha='center', va='center', color='white')
            else:
                ax2.text(j, i, policy_matrix[i, j], ha='center', va='center', color='white')
    
    # Mark start and goal
    ax2.text(0, CLIFF_ROW, 'S', ha='center', va='center', color='green', fontweight='bold')
    ax2.text(grid_shape[1]-1, CLIFF_ROW, 'G', ha='center', va='center', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Use the function
trained = q_learning()
plot_q_table(trained)