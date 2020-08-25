import numpy as np
from src.interact_with_matrices import open_object

def compute_degree_centrality(degree_matrix,
                              keys):
    """
        Computes the degree centrality of the nodes
        in the graph

        Arguments
        ---------
        degree_matrix (np.array): the degree matrix
        keys (list): contains the words

        Returns
        -------
        degree_centrality (dict): contains the degree centraliy 
        keyed by words
    """
    degrees = np.diag(degree_matrix)
    degree_centrality = degrees / (degree_matrix.shape[0] - 1)

    return {k:v for k,v in zip(keys, degree_centrality)}

def compute_betweennes