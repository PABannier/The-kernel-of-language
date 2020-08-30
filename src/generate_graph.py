import numpy as np
import joblib
from tqdm import tqdm
import string
import gc
import re

import h5py

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stop_words')
nltk.download('wordnet')

STOP_WORDS = set(stopwords.words('french'))

ADJACENCY_OUTFILE = '../data/french_adjacency_matrix.hdf5'
DEGREE_OUTFILE = '../data/french_degree_matrix.hdf5'
LAPLACIAN_OUTFILE = '../data/french_laplacian_matrix.hdf5'
KEYS_OUTFILE = '../data/french_keys.pkl'

DICT_INFILE = '../data/22000_words_list.pkl'

ACCEPTABLE_POS = [
    'JJ', 
    'JJR',
    'JJS',
    'NN', 
    'NNS', 
    'VB',
    'VBG', 
    'VBN',
    'VBP',
    'VBZ',
]

VERBAL_POS = [
    'VB', 
    'VBG',
    'VBN',
    'VBP',
    'VBZ'
]

def extract_tokens_from_definition(definition, word):
    """
        Tokenize the word definition by stripping punctuation,
        tokenizing, removing the word from its actual definition,
        removing stop words, keeping only nouns, adjectives and adverbs.

        Arguments
        ---------
        definition (string): the definition of the word
        word (string): the word

        Returns
        -------
        tokens (list): contains the tokens of the definition
    """

    # Remove words between parenthesis
    definition = str(definition)
    definition = re.sub('\(.*\)', '', definition)

    # Remove punctuation and make string lowercase
    definition = definition.translate(definition.maketrans('', '', string.punctuation)) 
    definition = definition.lower() 

    # Tokenization and removing stop words
    tokens = nltk.word_tokenize(definition)  
    tokens = [x for x in tokens if x is not word] 
    tokens = [x for x in tokens if x not in STOP_WORDS]

    # POS tagging
    tagged = nltk.pos_tag(tokens)
    tokens = [x for x in tagged if x[1] in ACCEPTABLE_POS]

    # Converting verbs to base form
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(x[0], 'v') if x[1] in VERBAL_POS
        else x[0]
        for x in tokens
    ]

    tokens = [
        lemmatizer.lemmatize(x) for x in tokens
    ]

    return tokens

def update_adjacency_matrix(keys, 
                            adjacency_matrix, 
                            definition, 
                            word,
                            word_index):
    """
        Updates the adjacency matrix by incrementing the coefficients
        by 1 for the words present in the definition of word

        Argument
        --------
        keys (list): list of keys of the English dictionary
        adjacency_matrix (np.array): adjacency matrix
        definition (string): definition of the word
        word (string): the word
        word_index (int): the index of the word in the keys list

        Returns
        -------
        adjacency_matrix (np.array): adjacency matrix
    """

    tokens = extract_tokens_from_definition(definition, word)

    for token in tokens:
        try:
            idx_token = keys.index(token)
        except ValueError:
            continue
        
        if word_index != idx_token:
            adjacency_matrix[word_index, idx_token] = 1

    return adjacency_matrix

def generate_degree_vector(adjacency_matrix):
    """
        Generates the degree matrix from the adjacency matrix
        by summing the columns of tha adjacency matrix and 
        projecting the results onto the diagonal

        Argument
        --------
        adjacency_matrix (np.array): the adjacency matrix

        Returns
        -------
        degree_matrix (np.array): the degree matrix
    """
    degrees = np.sum(adjacency_matrix, axis=0)
    return degrees

def generate_laplacian_matrix(adjacency_matrix, degree_matrix):
    """
        Generates the Laplacian matrix from the adjacency and degree
        matrices
        Recall: M(l) = M(d) - M(a)

        Argument
        --------
        adjacency_matrix (np.array): the adjacency matrix
        degree_matrix (np.array): the degree matrix

        Returns
        -------
        laplacian_matrix (np.array): the Laplacian matrix
    """
    return degree_matrix - adjacency_matrix

def save_obj(obj, outpath):
    """
        Saves obj in h5 format at the outpath adress

        Arguments
        ---------
        obj (np.array): the matrix to save
        outpath (string): the path of output file
    """

    with h5py.File(outpath, 'w') as outfile:
        outfile.create_dataset('matrix', data=obj)
    
    print('Successfully saved matrix:', outpath)


if __name__ == '__main__':
    print('Opening English dictionary...')
    with open(DICT_INFILE, 'rb') as infile:
        english_dictionary = joblib.load(infile)
    
    print('Generate dictionary keys...')
    keys = list(english_dictionary.keys())

    print('Generating adjacency matrix...')
    voc_size = len(english_dictionary)
    adjacency_matrix = np.zeros((voc_size, voc_size))
    
    for i, (word, definition) in tqdm(enumerate(english_dictionary.items()), total=voc_size):
        if word not in list(string.ascii_lowercase):
            adjacency_matrix = update_adjacency_matrix(
                keys=keys,
                adjacency_matrix=adjacency_matrix,
                definition=definition,
                word=word,
                word_index=i
            )
    
    print('Matrix shape:', adjacency_matrix.shape)
        
    print('Generating degree matrix...')
    degree_vector = generate_degree_vector(adjacency_matrix)
    degree_matrix = np.diag(degree_vector)

    print('Generating Laplacian matrix...')
    laplacian_matrix = generate_laplacian_matrix(
        adjacency_matrix, 
        degree_matrix
    )
    
    save_obj(adjacency_matrix, ADJACENCY_OUTFILE)
    save_obj(degree_matrix, DEGREE_OUTFILE)
    save_obj(laplacian_matrix, LAPLACIAN_OUTFILE)

    with open(KEYS_OUTFILE, 'wb') as infile:
        joblib.dump(keys, infile) 
        print('Successfully saved the keys dictionary.')