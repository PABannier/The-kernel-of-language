import numpy as np 
import matplotlib.pyplot as plt
import joblib
import h5py

def reduce_dimension_adjacency_matrix(adjacency_matrix, 
                                      degree_matrix, 
                                      keys, 
                                      threshold=2):
    """
        Reduces the dimension of the adjacency matrix

        Arguments
        ---------
        adjacency_matrix (np.array): the adjacency matrix
        degree_vector (np.array): the degree vector
        keys (list): contains the words in the dictionary
        threshold (int): if degree of node < threshold, the node is removed

        Returns
        -------
        adjacency_matrix (np.array): the reduce-dimensioned adjacency matrix
        degree_matrix (np.array): the reduce-dimension degree matrix
    """
    degree_vector = np.diag(degree_matrix)

    # Remove from degree matrix rows with less than threshold
    idx_to_keep = degree_vector >= threshold
    idx_to_remove = degree_vector < threshold
    idx_to_remove = list(idx_to_remove)
    idx_to_remove = [i for i in range(len(idx_to_remove)) if idx_to_remove[i] == True]

    degree_matrix = np.diag(degree_vector[idx_to_keep])

    # Remove the same rows/columns on the adjacency matrix
    adjacency_matrix = np.delete(
        arr=np.delete(
            arr=adjacency_matrix,
            obj=idx_to_remove,
            axis=1
        ),
        obj=idx_to_remove,
        axis=0
    )

    # Check the matrices are still square and have equal shape
    assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "The adjacency matrix is not square"
    assert degree_matrix.shape[0] == degree_matrix.shape[1], "The degree matrix is not square"
    assert degree_matrix.shape[0] == adjacency_matrix.shape[0], "Adjacency and degree matrices don't have the same shape"

    # Remove the indexes from the keys list
    keys = [keys[i] for i in range(len(keys)) if keys[i] not in idx_to_remove]

    return adjacency_matrix, degree_matrix, keys

def plot_vocabulary_size_against_threshold(adjacency_matrix,
                                           degree_matrix,
                                           keys,
                                           threshold_range=range(1, 20)):
    """
        Plots a regression graph of vocabulary size against the selected 
        threshold

        Arguments
        ---------
        adjacency_matrix (np.array): the adjacency matrix
        degree_matrix (np.array): the degree matrix
        keys (list): list of words
        threshold_range (list): list of threshold to test
    """

    shapes = []

    for threshold in threshold_range:
        matrix_tmp, _, _ = reduce_dimension_adjacency_matrix(
            adjacency_matrix=adjacency_matrix,
            degree_matrix=degree_matrix,
            keys=keys,
            threshold=threshold
        )
        shapes.append(matrix_tmp.shape[0])

    plt.plot(
        threshold_range, 
        shapes, 
        color='green', 
        marker='o', 
        linestyle='dashed', 
        linewidth=2, 
        markersize=12
    )
    plt.show()


if __name__ == '__main__':
    # Read objects
    adjacency_matrix = open_object('../data/adjacency_matrix.hdf5')
    degree_matrix = open_object('../data/degree_matrix.hdf5')
    laplacian_matrix = open_object('../data/laplacian_matrix.hdf5')
    
    with open('../data/keys.pkl', 'rb') as infile:
        keys = joblib.load(infile)

    plot_vocabulary_size_against_threshold(
        adjacency_matrix,
        degree_matrix,
        keys,
    )