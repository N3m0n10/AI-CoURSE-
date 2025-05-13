import numpy as np
from Project_2.q2_value_iter import *
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
    
    # Create plot
    plt.plot()
    
    # Plot policy arrows
    plt.imshow(value_matrix)   # ,interpolation="bilinear"
    plt.title('Optimal Policy')
    
    # Add arrows for policy
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if (i == CLIFF_ROW and 0 < j < grid_shape[1]-1):
                plt.text(j, i, 'X', ha='center', va='center', color='white')
            else:
                plt.text(j, i, f'\n\n {(value_matrix[i,j]):.2f}', ha='center', va='center', color='white')
                plt.text(j, i, policy_matrix[i, j], ha='center', va='center', color='white')
    
    # Mark goal 
    plt.text(grid_shape[1]-1, CLIFF_ROW, 'G', ha='center', va='center', color='red', fontweight='bold')

    plt.show()

# Use the function
trained = q_learning()
plot_q_table(trained)