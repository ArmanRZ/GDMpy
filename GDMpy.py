# By: Arman Rahimzamani from University of Washington
# Email: armanrz@uw.edu

import numpy as np
from copy import deepcopy
import scipy.spatial as ss
from scipy.special import digamma, gamma

########################################################################################################################
############################################## GRAPH DIVERGENCE MEASURE ################################################
########################################################################################################################


def gdm(data, graph, k):
    '''
    This function calculates the graph divergence measure from the observed samples samples. Note that this
    implementation assumes each node being a one-dimensional variable. In general, the variables need not be
    one-dimensional for GDM to exist.

    :param data: an N-by-d array containing N samples of a d-dimensional space
    :param graph: Hypothetical graphical model. A list of lists. The ith element is the list of parents of the ith node.
           Note that the total length has to be d.
    :param k: The k parameter of KNN

    :return: gdm value
    '''

    assert len(graph)==data.shape[1], "The dimension of data and the graph mismatch."
    N, d = data.shape
    res = N*np.sum([1 for parent in graph if len(parent)==0])*np.log(N) - N*np.log(N)

    # STEP 1: QUERY
    data_tree = ss.cKDTree(data)
    knn_dis = [data_tree.query(point, k + 1, p=np.inf)[0][k] for point in data] # Distance to the kth nearest neighbor for each point
    for i in range(N): res += digamma(len(data_tree.query_ball_point(data[i], knn_dis[i], p=np.inf)) -1 )

    # STEP 2: INQUIRE and COMPUTE
    for node_id in range(d):
        parents = graph[node_id]
        if len(parents) != 0:
            parents_tree = ss.cKDTree(data[:, np.sort(parents)])
            for i in range(N): res += np.log( len(parents_tree.query_ball_point(data[i,np.sort(parents)], knn_dis[i], p=np.inf)) )
        parents_plus_tree = ss.cKDTree(data[:, np.sort(parents+[node_id])])
        for i in range(N): res += \
            - np.log( len(parents_plus_tree.query_ball_point(data[i,np.sort(parents+[node_id])], knn_dis[i], p=np.inf)) )

    return res/N