import numpy as np
import h5py
import joblib

def print_k_most_frequent_words(degree_matrix, 
                                keys,
                                k=20):
    """
        Prints the k most frequent words from the adjacency matrix

        Arguments
        ---------
        degree_matrix (np.array): the degree matrix
        keys (list): the list of words
        k (integer): k
    """
    degrees = np.diag(degree_matrix)
    indices = list(np.argsort(-degrees))[:k]
    k_most = [(keys[i], int(degrees[i])) for i in range(len(keys)) if i in indices]
    print(f'{k} most frequent words', k_most)

def print_k_least_frequent_words(degree_matrix, 
                                 keys,
                                 k=20):
    """
        Prints the k least frequent from the adjacency matrix

        Arguments
        ---------
        degree_matrix (np.array): the degree matrix
        keys (list): the list of words
        k (integer): k
    """
    degrees = np.diag(degree_matrix)
    indices = list(np.argsort(degrees))[:k]
    k_least = [(keys[i], int(degrees[i])) for i in range(len(keys)) if i in indices]
    print(f'{k} least frequent words', k_least)

def print_k_words_for_threshold(degree_matrix, 
                                keys,
                                k=20,
                                threshold=5):
    """
        Prints k words whose degree is equal to threshold

        Arguments
        ---------
        adjacency_matrix (np.array): the adjacency matrix
        keys (list): the list of words
        k (integer): k
        threshold (integer): the threshold frequency
    """
    degrees = np.diag(degree_matrix)
    indices = np.argwhere(degrees == threshold)
    indices = list(indices)[:k]
    words = [keys[i] for i in range(len(keys)) if i in indices]
    print(f'{k} words above threshold', words)

def find_degree_for_word(degree_matrix,
                         keys,
                         word):
    """
    """
    degrees = np.diag(degree_matrix)

    try:
        idx = keys.index(word)
    except IndexError:
        print(f"{word} is not found in dictionary")
    
    degree = int(degrees[idx])
    print(f"Degree for {word}: {degree}")

def open_object(path):
    """
        Opens object saved with h5 format at path

        Argument
        ---------
        path (string): the path where the object is stored

        Returns
        -------
        obj (dict/np.array): the read object
    """

    with h5py.File(path, 'r') as infile:
        obj = infile['matrix'].value
    
    return obj


if __name__ == '__main__':
    # Read objects
    adjacency_matrix = open_object('../data/adjacency_matrix.hdf5')
    degree_matrix = open_object('../data/degree_matrix.hdf5')
    laplacian_matrix = open_object('../data/laplacian_matrix.hdf5')
    
    with open('../data/keys.pkl', 'rb') as infile:
        keys = joblib.load(infile)
    
    #find_degree_for_word(degree_matrix, keys, 'bacteria')
    
    """
    find_degree_for_word(degree_matrix, keys, 'animal')
    print('\n')
    
    print_k_most_frequent_words(
        degree_matrix, 
        keys, 
        k=50
    )
    print('\n')

    print_k_least_frequent_words(
        degree_matrix, 
        keys, 
        k=50
    )
    print('\n')
    """
    
    """
    print_k_words_for_threshold(
        degree_matrix, 
        keys, 
        k=30,
        threshold=30)

    print('Reducing dimension of adjacency matrix...')
    adjacency_matrix, degree_matrix, keys = reduce_dimension_adjacency_matrix(
        adjacency_matrix, 
        degree_matrix,
        keys,
        threshold=5
    )

    print('Matrix shape:', adjacency_matrix.shape)
    """


    plot_vocabulary_size_against_threshold(
        adjacency_matrix,
        degree_matrix,
        keys,
    )
    
    
