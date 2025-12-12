import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
from matplotlib.animation import FuncAnimation

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize convolution process')
parser.add_argument('-s', '--step', type=int, default=16,
                    help='Number of steps to show (default: 16 for full 4x4 output)')
args = parser.parse_args()

# Set random seed for reproducibility
np.random.seed(42)

# Generate random binary input matrix (6x6)
input_matrix = np.random.randint(0, 2, size=(6, 6))

# Define kernel (filter) - 3x3
kernel_matrix = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])

# Calculate all convolution positions with stride=1
# For 6x6 input and 3x3 kernel, output is 4x4
output_size = 4
stride = 1

# Generate all positions (row by row sweep)
positions = []
for out_row in range(output_size):
    for out_col in range(output_size):
        in_row = out_row * stride
        in_col = out_col * stride
        positions.append((in_row, in_col, out_row, out_col))

# Limit to requested number of steps
num_steps = min(args.step, len(positions))
positions = positions[:num_steps]

# Calculate output values for all positions
output_matrix = np.zeros((output_size, output_size), dtype=int)
for in_row, in_col, out_row, out_col in positions:
    patch = input_matrix[in_row:in_row+3, in_col:in_col+3]
    output_matrix[out_row, out_col] = np.sum(patch * kernel_matrix)

# Function to draw a matrix with grid
def draw_matrix(ax, matrix, x_start, y_start, cell_size=0.4, highlight=None,
                title="", title_offset=0.7, bg_color='white', show_values=None):
    rows, cols = matrix.shape

    for i in range(rows):
        for j in range(cols):
            # Determine cell color
            if highlight is not None and (i, j) in highlight:
                facecolor = 'lightblue'
            elif bg_color != 'white':
                facecolor = bg_color
            else:
                facecolor = 'white'

            # Draw cell
            rect = patches.Rectangle((x_start + j*cell_size, y_start - i*cell_size),
                                     cell_size, cell_size,
                                     linewidth=1.5, edgecolor='black',
                                     facecolor=facecolor)
            ax.add_patch(rect)

            # Add text - either show specific values or all values
            if show_values is not None:
                if (i, j) in show_values:
                    ax.text(x_start + j*cell_size + cell_size/2,
                           y_start - i*cell_size + cell_size/2,
                           str(matrix[i, j]),
                           ha='center', va='center', fontsize=11, fontweight='bold')
            else:
                ax.text(x_start + j*cell_size + cell_size/2,
                       y_start - i*cell_size + cell_size/2,
                       str(matrix[i, j]),
                       ha='center', va='center', fontsize=11, fontweight='bold')

    # Add title
    if title:
        ax.text(x_start + (cols*cell_size)/2, y_start + title_offset,
               title, ha='center', va='center', fontsize=12, fontweight='bold')

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 5))

def init():
    ax.clear()
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')
    return []

def update(frame):
    ax.clear()
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')

    # Get current position
    in_row, in_col, out_row, out_col = positions[frame]

    # Define highlight cells for current position
    highlight_cells = [(in_row + i, in_col + j) for i in range(3) for j in range(3)]

    # Draw input matrix with highlighted region
    x_input, y_input = 0.5, 4.5
    draw_matrix(ax, input_matrix, x_input, y_input, cell_size=0.4,
               highlight=highlight_cells, title="Input", title_offset=0.7)

    # Extract and draw current patch
    patch_matrix = input_matrix[in_row:in_row+3, in_col:in_col+3]
    x_patch, y_patch = 5.0, 3.7
    draw_matrix(ax, patch_matrix, x_patch, y_patch, cell_size=0.4,
               title="Image patch", title_offset=0.7)

    # Draw multiplication symbol
    ax.text(6.8, 3.5, "*", ha='center', va='center', fontsize=24, fontweight='bold')

    # Draw kernel with light salmon background
    x_kernel, y_kernel = 7.5, 3.7
    draw_matrix(ax, kernel_matrix, x_kernel, y_kernel, cell_size=0.4,
               title="Filter", title_offset=0.7, bg_color='lightsalmon')

    # Draw output matrix with values computed so far
    x_output, y_output = 11.0, 4.1
    cell_size = 0.4

    # Show values that have been computed up to current frame
    show_values = set()
    for f in range(frame + 1):
        _, _, o_r, o_c = positions[f]
        show_values.add((o_r, o_c))

    draw_matrix(ax, output_matrix, x_output, y_output, cell_size=0.4,
               title="Output", title_offset=0.7, show_values=show_values)

    # Add step information
    ax.text(8, 0.5, f'Step {frame + 1}/{num_steps}',
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return []

# Create animation
anim = FuncAnimation(fig, update, init_func=init, frames=num_steps,
                     interval=800, repeat=True, blit=False)

anim.save("conv.gif", writer="imagemagick", fps=1)

plt.tight_layout()
plt.show()
