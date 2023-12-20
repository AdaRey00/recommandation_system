from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def pairwise_cosine_similarity(matrix, top_n=10, chunk_size=100):
    n_rows = matrix.shape[0]

    for start_row in range(0, n_rows, chunk_size):
        end_row = min(start_row + chunk_size, n_rows)
        chunk = matrix[start_row:end_row]

        for comparison_start in range(0, n_rows, chunk_size):
            comparison_end = min(comparison_start + chunk_size, n_rows)
            comparison_chunk = matrix[comparison_start:comparison_end]

            # Compute cosine similarity for the batch
            chunk_similarities = cosine_similarity(chunk, comparison_chunk)

            for i, row_idx in enumerate(range(start_row, end_row)):
                # Get indices of top N similar books, excluding the book itself
                if comparison_start <= row_idx < comparison_end:
                    # Exclude self-similarity from the results
                    top_indices = np.argsort(chunk_similarities[i])[:-top_n-1:-1]
                else:
                    top_indices = np.argsort(chunk_similarities[i])[-top_n:]

                # Yield the row index and the indices of the top N similar rows
                yield row_idx, top_indices


def hybrid_recommendations(user_id, book_id, top_n=5):
    # Validate user_id and book_id
    if user_id not in interaction_matrix.index or book_id not in isbn_to_id:
        raise ValueError("User ID or Book ID not found in the dataset")

    # Collaborative filtering predictions
    user_idx = user_id - 1  # Convert to 0-based index
    predicted_ratings = np.dot(np.dot(U[user_idx, :], sigma), Vt)

    # Content-based filtering using book titles
    title_idx = isbn_to_id[book_id]

    # Initialize an array for title similarity scores
    title_sim_scores = np.zeros(len(id_to_isbn))

    # Get similarity scores for the specific book from the dictionary
    if title_idx in cosine_sim_titles:
        for other_idx, sim_score in cosine_sim_titles[title_idx].items():
            title_sim_scores[other_idx] = sim_score

    # Combine collaborative and content-based scores
    hybrid_scores = 0.7 * predicted_ratings + 0.3 * title_sim_scores

    # Get top N book IDs based on hybrid scores (excluding already rated books)
    rated_books = interaction_matrix.loc[user_id]
    unrated_books = np.isnan(rated_books)
    hybrid_scores[~unrated_books] = -np.inf  # Exclude rated books from recommendations
    top_book_indices = np.argsort(hybrid_scores)[::-1][:top_n]

    # Convert book indices back to ISBNs
    top_book_isbns = [id_to_isbn[idx] for idx in top_book_indices]

    return top_book_isbns
