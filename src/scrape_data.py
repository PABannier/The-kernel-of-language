import requests
import string
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import joblib
import os

OUTPATH = '../data/english_dictionary.pkl'
WORD_LIST = os.path.join('..', 'data/10000_words.txt')

def read_10000_word_list():
    """
        Reads the 10,000 word list
        and returns the words in a Python list

        Returns
        -------
        words (list): the 10,000 word list
    """

    with open(WORD_LIST, 'r') as infile:
        word_list = [line.rstrip('\n') for line in infile]
    
    return word_list

def get_url_from_letter(letter):
    """
        Gets the URL adress from the letter

        Argument
        --------
        letter (string): letter from the alphabet

        Returns
        -------
        url (string): url adress

    """
    return f"http://www.mso.anu.edu.au/~ralph/OPTED/v003/wb1913_{letter}.html"

def create_request_for_page(url):
    """
        Creates a request to retrieve a HTML document

        Argument
        --------
        url (string): url adress

        Returns
        -------
        page (string): HTML document containing a list of words and 
                       their definition
    """
    page = requests.get(url)
    assert page.status_code == 200, f"Failed loading: {url} "

    return page.text

def get_list_of_words(page):
    """
        Gets the list of all the words from a page in HTML format

        Argument
        --------
        page (string): HTML document containing a list of words and
                       their defintion

        Returns
        -------
        definitions (list): list of words (word + definition)
    """
    soup = BeautifulSoup(page, 'html.parser')
    return soup.find_all('p')

def extract_word_and_definition_from_tag(tag):
    """
        Extracts a word and its definition from a paragraph tag

        Argument
        --------
        tag (string): HTML tag containing a word and its definition

        Returns
        -------
        word (string): the word
        definition (string): its definition
    """
    word = tag.find('b').contents[0]
    word = word.lstrip()
    word = word.lower()

    definition = tag.get_text()
    definition = definition.replace('()', '')
    definition = definition.lstrip()
    definition = definition.lower()

    return word, definition

    
def generate_dictionary(tag_list, word_list):
    """
        Generates a dictionary from a list of tags

        Argument
        --------
        tag_list (list): list containing all the <p> tags
        word_list (list): list containing the 10,000 words

        Returns
        -------
        letter_dict (dictionary): key-value pairs with keys being the words
                                  and value their definition
    """
    letter_dict = dict()

    for tag in tqdm(tag_list, total=len(tag_list), desc='Tags'):
        word, definition = extract_word_and_definition_from_tag(tag)
        if (word not in letter_dict) and (word in word_list):
            word = word.lower()
            letter_dict[word] = definition
    
    return letter_dict


if __name__ == '__main__':
    # Initializing the English dictionary
    print('Initiliazing English dictionary...')
    english_dict = dict()

    print('Loading 10,000 word list...')
    word_list = read_10000_word_list()

    print('Generating letter dictionaries...')
    # Generating letter dictionaries
    for letter in tqdm(string.ascii_lowercase, total=26, desc='Letters'):
        url = get_url_from_letter(letter)
        page = create_request_for_page(url)
        tag_list = get_list_of_words(page)
        letter_dict = generate_dictionary(tag_list, word_list)
        english_dict.update(letter_dict)
    
    print('Number of words in dictionary:', len(list(english_dict.keys())))
    
    print('Saving the English dictionary...')
    # Pickling the English dictionary
    with open(OUTPATH, 'wb') as outfile:
        joblib.dump(english_dict, outfile)
    
