import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from flask import Flask
from flask import render_template, request
import pandas as pd

from analysis import header_analysis


app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def compare():
    """Example Hello World route."""



    df = None
    url, phrase, error = None, None, None
    # If request has been made process similarity
    if request.method == 'POST':
        url = request.form['url']
        phrase = request.form['phrase']
        df = header_analysis(url, phrase)
    return render_template('index.html', df=df, url=url, phrase=phrase, error=error)





if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))