import pandas as pd
import csv
from sklearn.cluster import KMeans


def k(df, clusters=2, random_state=40):
    # Here we set up a generic KMeans function
    kmeans = KMeans(n_clusters=clusters, random_state=random_state)
    clustering = kmeans.fit(df[['revenue', 'recency', 'frequency']])

    return clustering.cluster_centers_


def distance(df, centroids):
    '''
    this function computes the distances for each customer to each centroid. The shortest distance is then deemed to
    be the customer's respective cluster. The  calculation used is the Euclidian distance.
    '''
    
    distances_list = []
    for centroid in centroids:
        distances = ((df['revenue'] - centroid[0])**2 + (df['recency'] - centroid[1])**2 + (df['frequency'] -
                                                                                            centroid[2])**2)**.5
        distances_list.append(distances)

    # The distances for each customer to each centroid is then converted into a Pandas dataframe and transposed.
    distance_df = pd.DataFrame(distances_list)
    distance_df = distance_df.transpose()

    return distance_df


def outer_scope_centroids(outer_scope_unclassified, detection_idx, num_of_clusters):
    outer_anchors = k(outer_scope_unclassified[detection_idx], clusters=num_of_clusters)
    # these centroids are then sorted based on recency.
    outer_anchors = outer_anchors[outer_anchors[:, 0].argsort()]

    return outer_anchors


def outer_scope_handler(outer_scope_unclassified, outer_anchors):
    # We then compute the distances from each customer to each centroid
    outer_scope_classified = distance(outer_scope_unclassified, outer_anchors)
    # We then create a new column that represents the customer's cluster membership based on the distances returned
    outer_scope_unclassified['outer_scope_group'] = outer_scope_classified.idxmin(axis="columns")

    # Note: this variable name is misleading - the customers returned here are assigned an outer scope cluster.

    return outer_scope_unclassified


''''
def elimination(outer_scope_classified):
    # we eliminate the outer scope clusters 1 and 2 in this step
    inner_scope_unclassified1 = outer_scope_classified[outer_scope_classified['outer_scope_group'] != 0].copy()
    outer_scope_dropped = outer_scope_classified[outer_scope_classified['outer_scope_group'] == 0].copy()
    outer_scope_dropped['group'] = outer_scope_dropped['outer_scope_group']

    return inner_scope_unclassified1, outer_scope_dropped
'''

def elimination(outer_scope_classified):
    # we eliminate the outer scope clusters 1 and 2 in this step
    inner_scope_unclassified1 = outer_scope_classified[~outer_scope_classified['outer_scope_group'].isin([0, 1])].copy()
    outer_scope_dropped = outer_scope_classified[outer_scope_classified['outer_scope_group'].isin([0, 1])].copy()
    outer_scope_dropped['group'] = outer_scope_dropped['outer_scope_group']

    return inner_scope_unclassified1, outer_scope_dropped


def inner_scope_centroids(inner_scope_unclassified, num_of_clusters):
    # here we do the same thing as with the outer, except using 3 clusters (which we got from our elbow plot)
    # we also obviously are only using the remaining cluster
    inner_anchors = k(inner_scope_unclassified, clusters=num_of_clusters)

    # We then sort these centroids based on revenue
    inner_anchors = inner_anchors[inner_anchors[:, 0].argsort()]

    return inner_anchors


def inner_scope_handler(inner_scope_unclassified, inner_anchors):
    # We then compute the distances from each customer to each centroid
    inner_scope_classified = distance(inner_scope_unclassified, inner_anchors)
    # We then create a new column that represents the customer's cluster membership based on the distances returned
    inner_scope_unclassified['group'] = (inner_scope_classified.idxmin(axis="columns") +2)

    # Note: this variable name is misleading - the customers returned here are assigned an inner scope cluster.

    return inner_scope_unclassified
