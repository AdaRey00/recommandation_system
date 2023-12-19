import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

book_df = pd.read_csv('../data/Books.csv', low_memory=False)
rating_df = pd.read_csv('../data/Ratings.csv')
users_df = pd.read_csv('../data/Users.csv')

df = book_df.merge(rating_df, on='ISBN').merge(users_df, on='User-ID')

# Create numeric isbn
unique_isbns = df['ISBN'].unique()
isbn_to_id = {isbn: idx for idx, isbn in enumerate(unique_isbns)}
id_to_isbn = {idx: isbn for isbn, idx in isbn_to_id.items()}
df['ISBN_numeric'] = df['ISBN'].map(isbn_to_id)

# Filter for books with a minimum number of ratings
min_book_ratings = 50
filter_books = df['ISBN_numeric'].value_counts() > min_book_ratings
filter_books = filter_books[filter_books].index.tolist()

# Filter for users with a minimum number of ratings
min_user_ratings = 50
filter_users = df['User-ID'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

# Apply filters to the DataFrame
df_filtered = df[(df['ISBN_numeric'].isin(filter_books)) & (df['User-ID'].isin(filter_users))]
print(len(df_filtered))

# Now create the interaction matrix
interaction_matrix = df_filtered.pivot(index='User-ID', columns='ISBN_numeric', values='Book-Rating').fillna(0)
# Create a sparse matrix
matrix_sparse = csr_matrix(interaction_matrix.values)

NUMBER_OF_FACTORS_MF = 15

# Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(matrix_sparse.toarray(), k=NUMBER_OF_FACTORS_MF)

sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Create a content-based feature matrix using book titles
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
book_titles_tfidf = tfidf_vectorizer.fit_transform(book_df['Book-Title'].fillna(''))


def pairwise_cosine_similarity(matrix, chunk_size=100):
    n_rows = matrix.shape[0]
    similarities = {}

    for start_row in range(0, n_rows, chunk_size):
        end_row = min(start_row + chunk_size, n_rows)
        chunk = matrix[start_row:end_row]

        for comparison_start in range(0, n_rows, chunk_size):
            comparison_end = min(comparison_start + chunk_size, n_rows)
            comparison_chunk = matrix[comparison_start:comparison_end]

            chunk_similarities = cosine_similarity(chunk, comparison_chunk)

            for i, row_idx in enumerate(range(start_row, end_row)):
                if row_idx not in similarities:
                    similarities[row_idx] = {}

                for j, comparison_idx in enumerate(range(comparison_start, comparison_end)):
                    if row_idx != comparison_idx:
                        similarities[row_idx][comparison_idx] = chunk_similarities[i, j]

    return similarities


# Calculate cosine similarity between book titles in batches
cosine_sim_titles = pairwise_cosine_similarity(book_titles_tfidf, chunk_size=10)


# Function to get top N book recommendations based on a hybrid approach
def hybrid_recommendations(user_id, book_id, top_n=5):
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


# Example usage: Recommend books for user_id 123 and book_id '0452282152'
recommended_books = hybrid_recommendations(user_id=123, book_id='0452282152', top_n=5)
print("Recommended Books:")
for i, isbn in enumerate(recommended_books, 1):
    print(f"{i}. ISBN: {isbn}")
