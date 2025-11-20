from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import LinearSegmentedColormap

# Load image
img = Image.open(sys.argv[1])
img_array = np.array(img)
# Extract red channel
red_channel = img_array[:, :, 0]

print(f"Red channel shape: {red_channel.shape}")

# Get dimensions
rows, cols = red_channel.shape

# Extract the parts we want to display
top_rows = red_channel[:5, :]
bottom_rows = red_channel[-2:, :]

# Now extract columns from these rows
top_left = top_rows[:, :5]
top_right = top_rows[:, -2:]
bottom_left = bottom_rows[:, :5]
bottom_right = bottom_rows[:, -2:]

# Create the display grid with placeholders
display_size = 5 + 1 + 2  # 5 first + 1 for "..." + 2 last = 8
display_grid = np.full((display_size, display_size), np.nan)

# Fill in the actual values
display_grid[:5, :5] = top_left
display_grid[:5, -2:] = top_right
display_grid[-2:, :5] = bottom_left
display_grid[-2:, -2:] = bottom_right

# Create the plot
fig, ax = plt.subplots(figsize=(12, 10))

# Create a red colormap
cmap = plt.cm.Reds

# Draw colored rectangles for cells with values
for i in range(display_size):
    for j in range(display_size):
        if i == 5 or j == 5:  # Ellipsis row/column
            # Draw white background for ellipsis
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                facecolor='white', edgecolor='black', linewidth=1)
            ax.add_patch(rect)
        elif not np.isnan(display_grid[i, j]):
            value = display_grid[i, j]
            # Normalize value to 0-1 range for colormap
            normalized_value = value / 255.0
            color = cmap(normalized_value)

            # Draw colored rectangle
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)

# Add text to each cell
for i in range(display_size):
    for j in range(display_size):
        if i == 5 or j == 5:  # Ellipsis row/column
            ax.text(j, i, '...', ha='center', va='center', fontsize=12, fontweight='bold')
        elif not np.isnan(display_grid[i, j]):
            value = display_grid[i, j]
            # Use white text for dark backgrounds, black for light backgrounds
            text_color = 'white' # if value > 127 else 'black'
            ax.text(j, i, f'{value:.0f}', ha='center', va='center',
                   fontsize=10, color=text_color, fontweight='bold')

ax.set_xlim(-0.5, display_size - 0.5)
ax.set_ylim(-0.5, display_size - 0.5)
ax.set_aspect('equal')
ax.invert_yaxis()

# Set tick labels
row_labels = [f'{i}' for i in range(5)] + ['...'] + [f'{rows-2}', f'{rows-1}']
col_labels = [f'{i}' for i in range(5)] + ['...'] + [f'{cols-2}', f'{cols-1}']

ax.set_xticks(range(display_size))
ax.set_yticks(range(display_size))
ax.set_xticklabels(col_labels)
ax.set_yticklabels(row_labels)

ax.set_xlabel('Column Index', fontsize=12)
ax.set_ylabel('Row Index', fontsize=12)
ax.set_title('Red Channel Values (First 5 and Last 2 rows/columns)', fontsize=14, fontweight='bold')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=255))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Red Channel Intensity', fontsize=12)

plt.tight_layout()
plt.show()
