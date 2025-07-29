import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans


def plot3d(df, month_number=0):
    # this function plots the graph in 3D without any colors
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(df['Revenue'], df['Recency'], df['Frequency'])

    fig.suptitle('Month ' + str(month_number))
    ax.set_xlabel('Revenue', fontsize=14)
    ax.set_ylabel('Recency', fontsize=14)
    ax.set_zlabel('Frequency', fontsize=14)

    plt.show()

def create_color_map(df, group_column):
    unique_labels = sorted(df[group_column].unique())
    palette = sns.color_palette("coolwarm", len(unique_labels))
    return dict(zip(unique_labels, palette))

color_map_inner = None
color_map_outer = None

def plot3d_clusters_inner(df, centroids=None, month_number=0, elev=10, azim=-40):
    """
    Plots inner scope clusters in a 3D scatter plot with consistent color mapping.
    """
    global color_map_inner
    if color_map_inner is None:
        color_map_inner = create_color_map(df, 'group')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(df['recency'], df['revenue'], df['frequency'], c=df['group'].map(color_map_inner), s=50)

    ax.set_xlabel('Recency', fontsize=13, labelpad=2)
    ax.set_ylabel('Monetary Value', fontsize=13, labelpad=2)
    ax.zaxis.set_label_position('lower')
    ax.set_zlabel('Frequency', fontsize=13, labelpad=-5)
    

    ax.set_xlim3d(1, 0)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)

    ax.view_init(elev=elev, azim=azim)  
    handles = [mpatches.Patch(color=color_map_inner[label], label=label) for label in sorted(color_map_inner.keys())]
    ax.legend(handles=handles, title="Clusters & Sub-Clusters", fontsize=12, title_fontsize=13)
    plt.show()


def plot3d_clusters_outer(df, centroids=None, month_number=0, elev=10, azim=-40):
    """
    Plots outer scope clusters in a 3D scatter plot with consistent color mapping.
    """
    global color_map_outer
    if color_map_outer is None:
        color_map_outer = create_color_map(df, 'outer_scope_group')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(df['recency'], df['revenue'], df['frequency'], c=df['outer_scope_group'].map(color_map_outer), s=50)

    ax.set_xlabel('Recency', fontsize=13, labelpad=2)
    ax.set_ylabel('Monetary Value', fontsize=13, labelpad=2)
    ax.zaxis.set_label_position('lower')
    ax.set_zlabel('Frequency', fontsize=13, labelpad=-5)
    
    ax.set_xlim3d(1, 0)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)

    ax.view_init(elev=elev, azim=azim) 
    handles = [mpatches.Patch(color=color_map_outer[label], label=label) for label in sorted(color_map_outer.keys())]
    ax.legend(handles=handles, title="Clusters", fontsize=12, title_fontsize=13)
    plt.show()

def plot3d_clusters_inner_color(dfs, centroids=None):
    # this function plots the outer scope clusters with colors that are consistent montt to month
    number_of_colors = dfs[0]["inner_scope_group"].nunique()
    color_labels = dfs[0]["inner_scope_group"].unique()
    palette = sns.color_palette(n_colors=number_of_colors)
    color_map = dict(zip(color_labels, palette))

    for x in range(0, len(dfs)):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(dfs[x]['Revenue'], dfs[x]['Recency'], dfs[x]['Frequency'],
                   c=dfs[x]['inner_scope_group'].map(color_map))

        fig.suptitle('Month ' + str(x+16))
        ax.set_xlabel('Revenue', fontsize=14)
        ax.set_ylabel('Recency', fontsize=14)
        ax.set_zlabel('Frequency', fontsize=14)
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)

        if centroids is not None:
            ax.text(centroids[0][0], centroids[0][1], centroids[0][2], 'Centroid 1')
            ax.text(centroids[1][0], centroids[1][1], centroids[1][2], 'Centroid 2')
            ax.text(centroids[2][0], centroids[2][1], centroids[2][2], 'Centroid 3')
            ax.text(centroids[3][0], centroids[3][1], centroids[3][2], 'Centroid 4')
            ax.text(centroids[4][0], centroids[4][1], centroids[4][2], 'Centroid 5')

        plt.show()


