import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
import palettable as colormap


def get_color(num):
    color = colormap.tableau.TrafficLight_9.hex_colors
    if num > len(color):
        raise ValueError(f'Invalid color num, must be between 0 and {len(color)}')
    return color[:num]


def get_marker(num):
    marker = [
        'o',
        'v',
        's',
        'p',
        '*',
        'h',
        'H',
        '+',
        'x',
        'D',
        'd',
        'P',
        'X',
        '1',
        '2',
        '3',
        '4',
        '8',
        '^',
        '<',
        '>',
    ]
    return marker[:num]


def plot_low_embedding(embedding, type='tsne'):
    '''

    @param embedding:
    @param type: the dimensionality reduction method , tsne or pca, default is tsne
    @return:
    '''

    if type == 'tsne':
        X = TSNE(
            n_components=2, random_state=33, perplexity=min(50, embedding.shape[0] - 1)
        ).fit_transform(embedding)
    elif type == 'tsne':
        X = PCA(n_components=2).fit_transform(embedding)
    else:
        raise NotImplementedError('the dimensionality reduction method no exist !!!!!!')
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    palette = sns.color_palette("bright", embedding.shape[0])
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=range(embedding.shape[0]),
        legend='full',
        palette=palette,
    )
    # plt.savefig('images/digits_tsne-pca.png', dpi=120)
    plt.show()


def get_graph_from_adj(adj: np.array, save_path=None):
    adj[np.abs(adj) < 0.5] = 0
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    nodes = nx.draw_networkx_nodes(
        G, pos=nx.spectral_layout(G), node_color='yellow', node_size=700
    )
    nodes.set_edgecolor('black')
    nodes.set_linewidth(2)
    edges = nx.draw_networkx_edges(
        G,
        pos=nx.spectral_layout(G),
        alpha=0.5,
        edgelist=[(u, v) for (u, v, d) in G.edges(data=True)],
        width=4,
        edge_color='black',
        arrowsize=20,
        arrowstyle='fancy',
    )
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path + 'node_graph.png')
    else:
        plt.show()
    plt.close()


def plot_hotmap(data, save_path):

    data[np.abs(data) < 0.9] = 0
    sns.heatmap(
        data, annot=True, fmt=".2f", linewidths=0.3, linecolor="grey", cmap="RdBu_r"
    )
    plt.savefig(save_path + 'hot_map.png')
    plt.close()


def plot_line(
    data_x: list,
    data_y: list,
    x_label: str,
    y_label: str,
    labels: list,
    path: str,
    is_fit=False,
    step=0.01,
):
    if len(labels) != data_y.shape[1]:
        raise ValueError('the length of labels and data_x must be the same')
    color = get_color(len(labels))
    marker = get_marker(len(labels))
    plt.figure(figsize=(8, 8), dpi=300)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.xlabel(
        x_label,
        fontweight='bold',
        fontsize=18,
        labelpad=6,
        fontproperties='Times New Roman',
    )
    plt.ylabel(
        y_label,
        fontweight='bold',
        fontsize=18,
        labelpad=6,
        fontproperties='Times New Roman',
    )
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)
    if is_fit:
        x = np.arange(min(data_x), max(data_x), step)
        for id in range(data_y.shape[1]):
            fit = np.polyfit(data_x, data_y[:, id], 3)
            y_fit = np.poly1d(fit)(x)
            plt.plot(
                x,
                y_fit,
                color=color[id],
                label=labels[id],
                linewidth=3,
                marker=marker[id],
                markersize=8,
                markevery=int(0.05 * len(y_fit)),
            )
    plt.legend(prop={'family': 'Times New Roman', 'size': 14}, frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    # plt.show()


def plot_causal_graph(graph, save_path):
    plot_hotmap(graph, save_path)
    # get_graph_from_adj(graph, save_path)
