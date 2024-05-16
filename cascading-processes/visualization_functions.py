import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

def plot_network(world, ax=None):
    """Plot the networkx graph for the given World object"""
    if ax is None:
        ax = __provide_missing_ax()
    net = world.network
    pos = nx.drawing.layout.fruchterman_reingold_layout(
        net
    )
    nodelist = list(net.nodes())
    raw_colors = [node.capacity-node.final_load if node.failed else node.capacity-node.load for node in world.schedule.agents]
    colors = [c if c>=0 else -1 for c in raw_colors]
    cmap = plt.cm.RdYlBu

    norm = mcolors.Normalize(vmin=min(colors), vmax=max(colors))
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map._A = []
    nx.draw_networkx(net, nodelist=nodelist, node_color=colors, pos=pos, ax=ax, cmap=cmap, vmin=-1, vmax=max(colors))
    
    if ax.figure is not None:
        cbar = ax.figure.colorbar(scalar_map, ax=ax)
        cbar.set_label("$z_i$: Net fragility")
        ax.axis("off")


def __provide_missing_ax():
    return plt.subplot(1, 1, 1)