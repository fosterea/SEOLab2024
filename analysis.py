import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import random

import textstat

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

import matplotlib.pyplot as plt
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# import collections
# import re

# from textblob import TextBlob

# from scipy.interpolate import lagrange
# import numpy as np
# import matplotlib.pyplot as plt

# from scipy.optimize import newton


def main():
    url = "https://github.com/"
    depth = 1  # Don't adjust this for recursion depth
    internal_links_details = get_internal_links_with_details(url, depth)
    for link_detail in internal_links_details:
        print(link_detail)

    # Randomly select some pages
    selected_pages = select_random_pages(internal_links_details, 20)
    for page in selected_pages:
        print(page)

    header_data_for_analysis = [] # Storing title and descriptions in the list for later analysis.

    # Calculate readability
    readability_scores = calculate_readability(selected_pages)
    for url, title, score in readability_scores:
        print(f"URL: {url}, Title: {title}, Flesch-Kincaid Score: {score}")
        header_data_for_analysis.append([url,score])

    # Ensure the necessary NLTK components are downloaded
    nltk.download('punkt')
    nltk.download('stopwords')

    url_analysis = analyze_urls(selected_pages)

    # Example usage
    print_analysis_results(url_analysis)

    # Example usage
    plot_keywords_network(url_analysis)

    # change the following phrase base on your website
    main_phrase = "best github page"
    similarity_scores = calculate_similarity(main_phrase, url_analysis)

    for i, item in enumerate(url_analysis):
        print(f"URL: {item['URL']}, Similarity Score: {similarity_scores[i]}")
        if len(header_data_for_analysis[0]) == 2: #making sure that we don't store duplicate vaule
            header_data_for_analysis[i].append(similarity_scores[i])

    readability_scores_only = extract_readability_scores(readability_scores)
    print(readability_scores_only)






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

def analyze_urls(urls_with_details):
    analysis_results = []

    for url, title, description in urls_with_details:
        title_phrases, title_keywords = extract_phrases_and_keywords(title)
        description_phrases, description_keywords = extract_phrases_and_keywords(description)

        analysis_results.append({
            'URL': url,
            'Title': title,
            'Description': description,
            'Title Phrases': title_phrases,
            'Title Keywords': title_keywords,
            'Description Phrases': description_phrases,
            'Description Keywords': description_keywords
        })

    return analysis_results

def print_analysis_results(analysis_results):
    for item in analysis_results:
        print(f"URL: {item['URL']}")
        print(f"Title: {item['Title']}")
        print("Title Phrases:", ', '.join(item['Title Phrases']))
        print("Title Keywords:", ', '.join(item['Title Keywords']))
        print(f"Description: {item['Description']}")
        print("Description Phrases:", ', '.join(item['Description Phrases']))
        print("Description Keywords:", ', '.join(item['Description Keywords']))
        print("-" * 100)  # Separator for readability

def plot_keywords_network(analysis_results):
    G = nx.Graph()

    for item in analysis_results:
        title_keywords = item['Title Phrases']
        description_keywords = item['Description Phrases']

        # Add nodes for each keyword
        for word in title_keywords:
            G.add_node(word, type='title', color='blue')

        for word in description_keywords:
            G.add_node(word, type='description', color='red')

        # Add edges between title and description keywords
        for title_word in title_keywords:
            for desc_word in description_keywords:
                G.add_edge(title_word, desc_word)
        break
    # Set node colors
    colors = [G.nodes[node]['color'] for node in G.nodes]

    # Draw the graph
    plt.figure(figsize=(12, 12))
    nx.draw(G, with_labels=True, node_color=colors, font_size=8, node_size=1500, edge_color='gray')
    plt.title("Network Graph of Title and Description Phrases")
    plt.show()

def calculate_similarity(main_phrase, titles_descriptions):
    # Preprocess and tokenize the texts
    stop_words = set(stopwords.words('english'))

    def tokenize(text):
        words = word_tokenize(text)
        return [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    # Combine the main phrase with titles and descriptions
    texts = [main_phrase] + [item['Title'] + " " + item['Description'] for item in titles_descriptions]
    #texts = [main_phrase] + [item['Description'] for item in titles_descriptions]
    # Compute TF-IDF representation
    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    return cosine_similarities

#extract readability scores from the readability's data
def extract_readability_scores(readability_data):
    # Extracting only the scores from the readability data
    scores = [item[2] for item in readability_data]  # assuming the score is the third element
    return scores


if __name__ == "__main__":
    main()