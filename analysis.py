import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import random

import textstat

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

# import matplotlib.pyplot as plt
# import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Ensure the necessary NLTK components are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# import collections
# import re

# from textblob import TextBlob

# from scipy.interpolate import lagrange
# import numpy as np
# import matplotlib.pyplot as plt

# from scipy.optimize import newton


def header_analysis(url, main_phrase):
    depth = 1  # Don't adjust this for recursion depth
    internal_links_details = get_internal_links_with_details(url, depth)
    # for link_detail in internal_links_details:
    #     print(link_detail)

    # Randomly select some pages
    selected_pages = select_random_pages(internal_links_details, 20)

    # Create data frame
    df = pd.DataFrame(selected_pages, columns =['URL', 'Title', 'Description'])
    # print(df.head(10))

    # extract key words
    df[['Title Phrases', 'Title Keywords']] = df['Title'].apply(lambda x: pd.Series(extract_phrases_and_keywords(x)))
    df[['Description Phrases', 'Description Keywords']] = df['Description'].apply(lambda x: pd.Series(extract_phrases_and_keywords(x)))

    # Calculate similarity
    df['Similarity ratio'] = calculate_similarity(main_phrase, df)

    # Calculate readability
    df['Readability'] = df.apply(lambda x: textstat.flesch_kincaid_grade(x.Title + " " + x.Description), axis=1)
    # print(df[['Similarity ratio']])

    return df






def get_page_details(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Get the title of the page
        title = soup.title.string if soup.title else 'No title'

        # Get the description of the page
        description_tag = soup.find("meta", attrs={"name": "description"})
        description = description_tag["content"] if description_tag else 'No description'

        return title, description

    except requests.RequestException:
        return 'No title', 'No description'
    
def get_internal_links_with_details(url, depth, visited=None):
    if visited is None:
        visited = set()

    if depth == 0 or url in visited:
        return []

    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        visited.add(url)

        links = soup.find_all('a', href=True)

        base_url_parsed = requests.utils.urlparse(url)
        base_url = '{uri.scheme}://{uri.netloc}'.format(uri=base_url_parsed)

        page_details = []

        for link in links:
            href = link['href']
            full_url = urljoin(base_url, href)
            if full_url.startswith(base_url) and full_url not in visited:
                title, description = get_page_details(full_url)
                page_details.append((full_url, title, description))
                visited.add(full_url)
                page_details.extend(get_internal_links_with_details(full_url, depth-1, visited))

        return page_details

    except requests.RequestException as e:
        print(f"Error fetching page: {e}")
        return []
    
def select_random_pages(pages, percentage):
    if not pages or percentage <= 0:
        return []

    number_of_pages = len(pages)
    number_to_select = max(1, int((percentage / 100) * number_of_pages))

    return random.sample(pages, number_to_select)

def calculate_readability(pages):
    readability_scores = []
    for url, title, description in pages:
        score = textstat.flesch_kincaid_grade(description)
        readability_scores.append((url, title, score))
    return readability_scores

def extract_phrases_and_keywords(text):
    stop_words = set(stopwords.words('english'))

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Extract phrases (n-grams) and keywords
    phrases = set()
    keywords = set()

    for sentence in sentences:
        # Tokenize the sentence into words
        words = word_tokenize(sentence)

        # Filter out stop words and single character words (like punctuation)
        filtered_words = [word for word in words if word.lower() not in stop_words and len(word) > 1]

        # Add filtered words to keywords
        keywords.update(filtered_words)

        # Create bigrams and trigrams from filtered words
        bigrams = ngrams(filtered_words, 2)
        trigrams = ngrams(filtered_words, 3)

        # Add bigrams and trigrams to phrases
        phrases.update([' '.join(gram) for gram in bigrams])
        phrases.update([' '.join(gram) for gram in trigrams])

    return list(phrases), list(keywords)




def calculate_similarity(main_phrase, df):
    # Preprocess and tokenize the texts
    stop_words = set(stopwords.words('english'))

    def tokenize(text):
        words = word_tokenize(text)
        return [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    
    texts = [main_phrase] + df['Title'].str.cat(df['Description'], sep=' ').tolist()
    
    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    return pd.Series(cosine_similarities)


# if __name__ == "__main__":
#     header_analysis('https://github.com', 'github')