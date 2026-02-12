"""
GraphSAGE Sampling Visualization
Demonstrates the layer-wise sampling process with both static and animated views
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from collections import defaultdict
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

pltstyle = "~/projects/packages/toolbox/plot/mystyle.mplstyle"
if os.path.exists(pltstyle):
    plt.style.use(pltstyle)

class GraphSAGEVisualizer:
    def __init__(self, num_nodes=100, avg_degree=8, batch_size=8, fanouts=[5, 3]):
        """
        Initialize the GraphSAGE visualizer

        Parameters:
        -----------
        num_nodes: int - Total number of nodes in the graph
        avg_degree: int - Average degree of nodes
        batch_size: int - Number of seed nodes to sample
        fanouts: list - Fanout for each layer [layer1_fanout, layer2_fanout]
        """
        self.num_nodes = num_nodes
        self.avg_degree = avg_degree
        self.batch_size = batch_size
        self.fanouts = fanouts

        # Create the graph
        self.G = self._create_graph()
        self.pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)

    def _create_graph(self):
        """Create a random graph with approximate average degree"""
        # Create a graph with power-law degree distribution (more realistic)
        G = nx.powerlaw_cluster_graph(self.num_nodes, int(self.avg_degree/2), 0.3, seed=42)
        # Make it undirected
        G = G.to_undirected()
        return G

    def sample_seeds(self):
        """Sample seed nodes for the batch"""
        return random.sample(list(self.G.nodes()), self.batch_size)

    def sample_neighbors(self, nodes, fanout):
        """
        Sample neighbors for given nodes with specified fanout

        Parameters:
        -----------
        nodes: list - List of nodes to sample neighbors from
        fanout: int - Number of neighbors to sample per node

        Returns:
        --------
        dict: mapping from node to its sampled neighbors
        """
        sampled = {}
        for node in nodes:
            neighbors = list(self.G.neighbors(node))
            if len(neighbors) > 0:
                # Sample with replacement if not enough neighbors
                k = min(fanout, len(neighbors))
                sampled[node] = random.sample(neighbors, k)
            else:
                sampled[node] = []
        return sampled

    def build_computation_graph(self, seeds):
        """
        Build the computation graph by sampling layer by layer

        Returns:
        --------
        dict: Contains seeds and sampled neighbors for each layer
        """
        result = {
            'seeds': seeds,
            'layer1': {},
            'layer2': {},
            'all_nodes': set(seeds)
        }

        # Layer 1 sampling (closest to seeds)
        result['layer1'] = self.sample_neighbors(seeds, self.fanouts[0])
        layer1_nodes = set()
        for neighbors in result['layer1'].values():
            layer1_nodes.update(neighbors)
        result['all_nodes'].update(layer1_nodes)

        # Layer 2 sampling (furthest from seeds)
        result['layer2'] = self.sample_neighbors(list(layer1_nodes), self.fanouts[1])
        layer2_nodes = set()
        for neighbors in result['layer2'].values():
            layer2_nodes.update(neighbors)
        result['all_nodes'].update(layer2_nodes)

        return result

    def create_static_visualization(self, seeds, comp_graph):
        """Create static visualization with 5 panels"""
        fig = plt.figure(figsize=(20, 12))

        # Panel 1: Full graph
        ax1 = plt.subplot(2, 3, 1)
        self._plot_full_graph(ax1)

        # Panel 2: Seeds highlighted
        ax2 = plt.subplot(2, 3, 2)
        self._plot_seeds(ax2, seeds)

        # Panel 3: Layer 1 sampling
        ax3 = plt.subplot(2, 3, 3)
        self._plot_layer1(ax3, seeds, comp_graph['layer1'])

        # Panel 4: Layer 2 sampling
        ax4 = plt.subplot(2, 3, 4)
        self._plot_layer2(ax4, seeds, comp_graph)

        # Panel 5: Final mini-graph
        ax5 = plt.subplot(2, 3, 5)
        self._plot_mini_graph(ax5, comp_graph)

        # Add information panel
        ax6 = plt.subplot(2, 3, 6)
        self._plot_info_panel(ax6, comp_graph)

        plt.tight_layout()
        return fig

    def _plot_full_graph(self, ax):
        """Plot the complete graph"""
        ax.set_title('1. Full Graph', fontsize=14, fontweight='bold')
        nx.draw_networkx_nodes(self.G, self.pos, node_color='lightblue',
                               node_size=100, alpha=0.6, ax=ax)
        nx.draw_networkx_edges(self.G, self.pos, alpha=0.2, ax=ax)
        ax.axis('off')
        ax.text(0.5, -0.1, f'Total nodes: {self.num_nodes}',
                ha='center', transform=ax.transAxes, fontsize=10)

    def _plot_seeds(self, ax, seeds):
        """Plot graph with seeds highlighted"""
        ax.set_title('2. Seed Nodes (Batch)', fontsize=14, fontweight='bold')

        # Draw all nodes
        node_colors = ['red' if n in seeds else 'lightgray' for n in self.G.nodes()]
        node_sizes = [300 if n in seeds else 50 for n in self.G.nodes()]

        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(self.G, self.pos, alpha=0.1, ax=ax)

        # Label seeds
        seed_labels = {n: str(n) for n in seeds}
        nx.draw_networkx_labels(self.G, self.pos, seed_labels,
                                font_size=8, font_color='white', ax=ax)

        ax.axis('off')
        ax.text(0.5, -0.1, f'Batch size: {len(seeds)}',
                ha='center', transform=ax.transAxes, fontsize=10)

    def _plot_layer1(self, ax, seeds, layer1_sampling):
        """Plot Layer 1 sampling"""
        ax.set_title(f'3. Layer 1 Sampling (Fanout={self.fanouts[0]})',
                     fontsize=14, fontweight='bold')

        # Get all layer 1 nodes
        layer1_nodes = set()
        for neighbors in layer1_sampling.values():
            layer1_nodes.update(neighbors)

        # Color nodes
        node_colors = []
        node_sizes = []
        for n in self.G.nodes():
            if n in seeds:
                node_colors.append('red')
                node_sizes.append(300)
            elif n in layer1_nodes:
                node_colors.append('orange')
                node_sizes.append(200)
            else:
                node_colors.append('lightgray')
                node_sizes.append(30)

        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.7, ax=ax)

        # Draw sampled edges
        sampled_edges = []
        for seed, neighbors in layer1_sampling.items():
            for neighbor in neighbors:
                sampled_edges.append((seed, neighbor))

        nx.draw_networkx_edges(self.G, self.pos, alpha=0.05, ax=ax)
        nx.draw_networkx_edges(self.G, self.pos, edgelist=sampled_edges,
                               edge_color='orange', width=2, alpha=0.8, ax=ax)

        ax.axis('off')
        ax.text(0.5, -0.1, f'Layer 1 nodes: {len(layer1_nodes)}',
                ha='center', transform=ax.transAxes, fontsize=10)

    def _plot_layer2(self, ax, seeds, comp_graph):
        """Plot Layer 2 sampling"""
        ax.set_title(f'4. Layer 2 Sampling (Fanout={self.fanouts[1]})',
                     fontsize=14, fontweight='bold')

        # Get layer nodes
        layer1_nodes = set()
        for neighbors in comp_graph['layer1'].values():
            layer1_nodes.update(neighbors)

        layer2_nodes = set()
        for neighbors in comp_graph['layer2'].values():
            layer2_nodes.update(neighbors)

        # Color nodes
        node_colors = []
        node_sizes = []
        for n in self.G.nodes():
            if n in seeds:
                node_colors.append('red')
                node_sizes.append(300)
            elif n in layer1_nodes:
                node_colors.append('orange')
                node_sizes.append(200)
            elif n in layer2_nodes:
                node_colors.append('yellow')
                node_sizes.append(150)
            else:
                node_colors.append('lightgray')
                node_sizes.append(30)

        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.7, ax=ax)

        # Draw all sampled edges
        sampled_edges = []
        # Layer 1 edges
        for seed, neighbors in comp_graph['layer1'].items():
            for neighbor in neighbors:
                sampled_edges.append((seed, neighbor))
        # Layer 2 edges
        for node, neighbors in comp_graph['layer2'].items():
            for neighbor in neighbors:
                sampled_edges.append((node, neighbor))

        nx.draw_networkx_edges(self.G, self.pos, alpha=0.05, ax=ax)
        nx.draw_networkx_edges(self.G, self.pos, edgelist=sampled_edges,
                               edge_color='green', width=2, alpha=0.6, ax=ax)

        ax.axis('off')
        ax.text(0.5, -0.1, f'Layer 2 nodes: {len(layer2_nodes)}',
                ha='center', transform=ax.transAxes, fontsize=10)

    def _plot_mini_graph(self, ax, comp_graph):
        """Plot the final mini-graph (computation graph)"""
        ax.set_title('5. Final Mini-Graph (Computation Graph)',
                     fontsize=14, fontweight='bold')

        # Create subgraph with only sampled nodes
        mini_nodes = list(comp_graph['all_nodes'])
        subG = self.G.subgraph(mini_nodes)

        # Get positions for mini-graph
        mini_pos = {n: self.pos[n] for n in mini_nodes}

        # Get layer nodes
        seeds = comp_graph['seeds']
        layer1_nodes = set()
        for neighbors in comp_graph['layer1'].values():
            layer1_nodes.update(neighbors)
        layer2_nodes = set()
        for neighbors in comp_graph['layer2'].values():
            layer2_nodes.update(neighbors)

        # Color nodes by layer
        node_colors = []
        for n in mini_nodes:
            if n in seeds:
                node_colors.append('red')
            elif n in layer1_nodes:
                node_colors.append('orange')
            else:
                node_colors.append('yellow')

        nx.draw_networkx_nodes(subG, mini_pos, node_color=node_colors,
                               node_size=300, alpha=0.8, ax=ax)

        # Draw sampled edges only
        sampled_edges = []
        for seed, neighbors in comp_graph['layer1'].items():
            for neighbor in neighbors:
                if subG.has_edge(seed, neighbor):
                    sampled_edges.append((seed, neighbor))
        for node, neighbors in comp_graph['layer2'].items():
            for neighbor in neighbors:
                if subG.has_edge(node, neighbor):
                    sampled_edges.append((node, neighbor))

        nx.draw_networkx_edges(subG, mini_pos, edgelist=sampled_edges,
                               edge_color='green', width=2, alpha=0.7, ax=ax)

        # Add labels
        nx.draw_networkx_labels(subG, mini_pos, font_size=7,
                                font_color='white', ax=ax)

        ax.axis('off')
        ax.text(0.5, -0.1, f'Total nodes in mini-graph: {len(mini_nodes)}',
                ha='center', transform=ax.transAxes, fontsize=10)

    def _plot_info_panel(self, ax, comp_graph):
        """Plot information panel with statistics"""
        ax.set_title('6. Sampling Statistics', fontsize=14, fontweight='bold')
        ax.axis('off')

        # Calculate statistics
        seeds = comp_graph['seeds']
        layer1_nodes = set()
        for neighbors in comp_graph['layer1'].values():
            layer1_nodes.update(neighbors)
        layer2_nodes = set()
        for neighbors in comp_graph['layer2'].values():
            layer2_nodes.update(neighbors)

        # Count edges
        layer1_edges = sum(len(neighbors) for neighbors in comp_graph['layer1'].values())
        layer2_edges = sum(len(neighbors) for neighbors in comp_graph['layer2'].values())

        info_text = f"""
        SAMPLING PROCESS SUMMARY
        {'='*40}

        Full Graph:
          • Total nodes: {self.num_nodes}
          • Total edges: {self.G.number_of_edges()}
          • Avg degree: {2*self.G.number_of_edges()/self.num_nodes:.1f}

        Sampling Configuration:
          • Batch size: {self.batch_size}
          • Layer 1 fanout: {self.fanouts[0]}
          • Layer 2 fanout: {self.fanouts[1]}

        Sampled Nodes:
          • Seeds: {len(seeds)}
          • Layer 1 neighbors: {len(layer1_nodes)}
          • Layer 2 neighbors: {len(layer2_nodes)}
          • Total unique: {len(comp_graph['all_nodes'])}

        Sampled Edges:
          • Layer 1: {layer1_edges}
          • Layer 2: {layer2_edges}
          • Total: {layer1_edges + layer2_edges}

        Complexity Reduction:
          • Without sampling: O(E) = O({self.G.number_of_edges()})
          • With sampling: O(N·S^L) = O({self.batch_size}·{self.fanouts[0]}·{self.fanouts[1]})
          • Reduction ratio: {self.G.number_of_edges() / (self.batch_size * self.fanouts[0] * self.fanouts[1]):.1f}x

        Color Legend:
          Red: Seed nodes
          Orange: Layer 1 neighbors
          Yellow: Layer 2 neighbors
        """

        ax.text(0.1, 0.95, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    def create_animation(self, num_frames=10, interval=1500):
        """Create animation showing different random samplings"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.subplots_adjust(top=0.88, bottom=0.08)

        def init():
            return []

        def update(frame):
            ax1.clear()
            ax2.clear()

            # Sample new seeds and build computation graph
            seeds = self.sample_seeds()
            comp_graph = self.build_computation_graph(seeds)

            # Left panel: Full graph with sampling highlighted
            ax1.set_title(f'Full Graph with Sampling (Frame {frame+1}/{num_frames})',
                          fontsize=14, fontweight='bold')

            # Get layer nodes
            layer1_nodes = set()
            for neighbors in comp_graph['layer1'].values():
                layer1_nodes.update(neighbors)
            layer2_nodes = set()
            for neighbors in comp_graph['layer2'].values():
                layer2_nodes.update(neighbors)

            # Color nodes
            node_colors = []
            node_sizes = []
            for n in self.G.nodes():
                if n in seeds:
                    node_colors.append('red')
                    node_sizes.append(300)
                elif n in layer1_nodes:
                    node_colors.append('orange')
                    node_sizes.append(200)
                elif n in layer2_nodes:
                    node_colors.append('yellow')
                    node_sizes.append(150)
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(30)

            nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors,
                                   node_size=node_sizes, alpha=0.7, ax=ax1)

            # Draw sampled edges
            sampled_edges = []
            for seed, neighbors in comp_graph['layer1'].items():
                for neighbor in neighbors:
                    sampled_edges.append((seed, neighbor))
            for node, neighbors in comp_graph['layer2'].items():
                for neighbor in neighbors:
                    sampled_edges.append((node, neighbor))

            nx.draw_networkx_edges(self.G, self.pos, alpha=0.05, ax=ax1)
            nx.draw_networkx_edges(self.G, self.pos, edgelist=sampled_edges,
                                   edge_color='green', width=2, alpha=0.6, ax=ax1)
            ax1.axis('off')

            # Right panel: Mini-graph
            ax2.set_title('Mini-Graph (Computation Graph)',
                          fontsize=14, fontweight='bold')

            mini_nodes = list(comp_graph['all_nodes'])
            subG = self.G.subgraph(mini_nodes)
            mini_pos = {n: self.pos[n] for n in mini_nodes}

            node_colors = []
            for n in mini_nodes:
                if n in seeds:
                    node_colors.append('red')
                elif n in layer1_nodes:
                    node_colors.append('orange')
                else:
                    node_colors.append('yellow')

            nx.draw_networkx_nodes(subG, mini_pos, node_color=node_colors,
                                   node_size=300, alpha=0.8, ax=ax2)

            nx.draw_networkx_edges(subG, mini_pos, edgelist=sampled_edges,
                                   edge_color='green', width=2, alpha=0.7, ax=ax2)

            nx.draw_networkx_labels(subG, mini_pos, font_size=7,
                                    font_color='white', ax=ax2)

            ax2.axis('off')
            ax2.text(0.5, -0.05,
                     f'Nodes: {len(mini_nodes)} | Seeds: {len(seeds)} | ' +
                     f'L1: {len(layer1_nodes)} | L2: {len(layer2_nodes)}',
                     ha='center', transform=ax2.transAxes, fontsize=10)

            return []

        anim = animation.FuncAnimation(fig, update, init_func=init,
                                       frames=num_frames, interval=interval,
                                       blit=False, repeat=True)

        # plt.tight_layout()
        return fig, anim


def main():
    """Main function to create visualizations"""

    # Create visualizer with smaller graph for clarity
    visualizer = GraphSAGEVisualizer(
        num_nodes=100,      # Total nodes in graph
        avg_degree=8,       # Average node degree
        batch_size=8,       # Number of seed nodes
        fanouts=[5, 3]      # [Layer1_fanout, Layer2_fanout]
    )

    # Sample seeds and build computation graph
    seeds = visualizer.sample_seeds()
    comp_graph = visualizer.build_computation_graph(seeds)

    # Create static visualization
    print("Creating static visualization...")
    fig_static = visualizer.create_static_visualization(seeds, comp_graph)
    plt.savefig('./graphsage_static_visualization.png',
                dpi=150, bbox_inches='tight')
    print("✓ Static visualization saved")

    # Create animation
    print("\nCreating animation...")
    fig_anim, anim = visualizer.create_animation(num_frames=10, interval=1500)

    # Save animation as GIF
    writer = animation.PillowWriter(fps=1, bitrate=1800)
    anim.save('./graphsage_animation.gif',
              writer=writer)
    print("✓ Animation saved")

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print("1. graphsage_static_visualization.png - 5-panel static view")
    print("2. graphsage_animation.gif - Animated random sampling")
    print("\nThe visualizations show:")
    print("• How GraphSAGE samples neighbors layer by layer")
    print("• The computational graph construction process")
    print("• Complexity reduction through sampling")
    print("• Different random samplings in the animation")


if __name__ == "__main__":
    main()
