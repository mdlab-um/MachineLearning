import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = Image.open(sys.argv[1])
img_array = np.array(img)

print(img_array.shape)

# Create colored channel visualizations
red_only = np.zeros_like(img_array)
red_only[:, :, 0] = img_array[:, :, 0]  # Only red channel active

green_only = np.zeros_like(img_array)
green_only[:, :, 1] = img_array[:, :, 1]  # Only green channel active

blue_only = np.zeros_like(img_array)
blue_only[:, :, 2] = img_array[:, :, 2]  # Only blue channel active

# Display in 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img_array)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(red_only)
axes[1].set_title('Red Channel (in color)')
axes[1].axis('off')

axes[2].imshow(green_only)
axes[2].set_title('Green Channel (in color)')
axes[2].axis('off')

axes[3].imshow(blue_only)
axes[3].set_title('Blue Channel (in color)')
axes[3].axis('off')

plt.tight_layout()
plt.show()
