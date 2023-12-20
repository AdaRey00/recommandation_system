from joblib import load
from flask import Flask, request, jsonify
from utils import hybrid_recommendations

# Load the saved components
tfidf_vectorizer = load('../models/tfidf_vectorizer.joblib')
cosine_sim_titles = load('../models/cosine_sim_titles.joblib')
interaction_matrix = load('models/interaction_matrix.joblib')
U = load('models/U.joblib')
sigma = load('models/sigma.joblib')
Vt = load('models/Vt.joblib')
isbn_to_id = load('models/isbn_to_id.joblib')
id_to_isbn = load('models/id_to_isbn.joblib')

app = Flask(__name__)


@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    book_ids = request.args.getlist('book_ids')  # List of book IDs
    top_n = request.args.get('top_n', default=5, type=int)
    recommendations = aggregate_recommendations(user_id, book_ids, top_n)
    return jsonify(recommendations)


def aggregate_recommendations(user_id, book_ids, top_n):
    combined_recommendations = {}

    # Get recommendations for each book and aggregate them
    for book_id in book_ids:
        try:
            recs = hybrid_recommendations(user_id, book_id, top_n)
            for rec in recs:
                if rec in combined_recommendations:
                    combined_recommendations[rec] += 1  # Increment if already present
                else:
                    combined_recommendations[rec] = 1
        except KeyError:
            # Handle cases where book_id is not found
            print(f"Book ID {book_id} not found in dataset")

    # Sort the aggregated recommendations by frequency and select the top N
    sorted_recs = sorted(combined_recommendations, key=combined_recommendations.get, reverse=True)

    # If fewer than top_n recommendations, just return them all
    return sorted_recs[:min(top_n, len(sorted_recs))]


if __name__ == '__main__':
    app.run(debug=True)
