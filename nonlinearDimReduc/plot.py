import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_colorlabels():
    # --- Load the encoded mammoth data (for color labels) ---
    with open("./data/mammoth_10k_encoded.json") as f:
        mammoth_encoded = json.load(f)

    label_offsets = mammoth_encoded["labelOffsets"]

    # --- Generate color indices exactly like in load-data.js ---
    n_points = sum(label_offsets)
    color_indices = np.zeros(n_points, dtype=int)

    color_index = -1
    n_color_points = 0
    for i in range(n_points):
        if i >= n_color_points:
            color_index += 1
            n_color_points += label_offsets[color_index]
        color_indices[i] = color_index
    # color_indices define the color index for each data point;
    return color_indices

def get_data():
    # --- Load 3D coordinates ---
    with open("./data/mammoth_3d.json") as f:
        mammoth_3d = json.load(f)

    # In the official repo, the 3D file is structured like:
    # { "points": [[x1, y1, z1], [x2, y2, z2], ...] }
    coords = np.array(mammoth_3d)
    return coords

def plot(coords, color_indices, projected_coords=None):

    if len(coords) != len(color_indices):
        raise ValueError(f"Mismatch: {len(coords)} coords vs {len(color_indices)} color indices")

    # Get a Matplotlib colormap and sample len(color_indices) distinct colors
    cmap = plt.get_cmap('tab20')  # or try 'tab10', 'Set3', 'Paired', etc.
    unique_labels = np.unique(color_indices)
    colors = [mcolors.to_hex(cmap(i / len(unique_labels))) for i in range(len(unique_labels))]

    # --- Plot in 3D ---
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter"}]],  # 3D left, 2D right
        subplot_titles=("3D Coordinates", "2D PCA Projection")
    )
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 2],
        z=coords[:, 1],
        mode='markers',
        marker=dict(size=2, color=color_indices,\
                colorscale=[[i/(len(colors)-1), c] for i, c in enumerate(colors)],
                opacity=0.8),
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=projected_coords[:, 0],
            y=projected_coords[:, 1],
            mode='markers',
            marker=dict(size=4, color=color_indices,\
                        colorscale=[[i/(len(colors)-1), c] for i, c in enumerate(colors)],
                        opacity=0.8),
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Z", row=1, col=1)
    fig.update_scenes(zaxis_title_text="Y", row=1, col=1)


    fig.update_xaxes(title_text="D1", row=1, col=2)
    fig.update_yaxes(title_text="D2", row=1, col=2)

    fig.show()

if __name__ == "__main__":
    coords = get_data()
    color_indices = get_colorlabels()

    # PCA
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    pca = PCA(n_components=2)
    # data_2d = pca.fit_transform(coords)  # shape will be (N, 2)
    data_2d = pca.fit_transform(coords_scaled)  # shape will be (N, 2)

    plot(coords, color_indices, projected_coords=data_2d)
