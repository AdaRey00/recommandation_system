import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from joblib import dump
from utils import pairwise_cosine_similarity
import os

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

# Create the interaction matrix
interaction_matrix = df_filtered.pivot(index='User-ID', columns='ISBN_numeric', values='Book-Rating').fillna(0)
# Create a sparse matrix
matrix_sparse = csr_matrix(interaction_matrix.values)

NUMBER_OF_FACTORS_MF = 15

# Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(matrix_sparse.toarray(), k=NUMBER_OF_FACTORS_MF)

sigma = np.diag(sigma)

# Create a content-based feature matrix using book titles
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
book_titles_tfidf = tfidf_vectorizer.fit_transform(book_df['Book-Title'].fillna(''))

# Calculate cosine similarity between book titles in batches
cosine_sim_titles = pairwise_cosine_similarity(book_titles_tfidf, chunk_size=10)

# Assuming your pairwise_cosine_similarity function yields row_idx, top_indices
cosine_similarity_dict = {}
for row_idx, top_indices in pairwise_cosine_similarity(book_titles_tfidf, chunk_size=10):
    cosine_similarity_dict[row_idx] = top_indices

# Now you can save this dictionary


# Check if the directory exists, and if not, create it
models_directory = '../models'
if not os.path.exists(models_directory):
    os.makedirs(models_directory)

# Save the necessary components
dump(tfidf_vectorizer, '../models/tfidf_vectorizer.joblib')
dump(cosine_similarity_dict, '../models/cosine_sim_titles.joblib')
dump(interaction_matrix, 'models/interaction_matrix.joblib')
dump(U, 'models/U.joblib')
dump(sigma, 'models/sigma.joblib')
dump(Vt, 'models/Vt.joblib')
dump(isbn_to_id, 'models/isbn_to_id.joblib')
dump(id_to_isbn, 'models/id_to_isbn.joblib')
