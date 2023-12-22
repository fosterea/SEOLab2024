import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from flask import Flask
from flask import render_template, request
import pandas


app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def compare():
    """Example Hello World route."""



    similarity_score = None
    article, phrase = None, None
    # If request has been made process similarity
    if request.method == 'POST':
        article = request.form['article']
        phrase = request.form['phrase']
        similarity_score = calculate_similarity(article, phrase)*100
    return render_template('compare.html', similarity_score=similarity_score, article=article, phrase=phrase)



def calculate_similarity(article, phrase):
    vectorizer = TfidfVectorizer()

    # Combine the article and phrase into one list for vectorization
    all_text = [article, phrase]

    # Convert the text to vectors
    tfidf_matrix = vectorizer.fit_transform(all_text)

    # Calculate cosine similarity (returns a matrix)
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # The similarity score between the article and the phrase
    similarity_score = similarity_matrix[0][0]

    return similarity_score


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))