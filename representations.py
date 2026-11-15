"""
representations.py - shared settings and corpus loader for Assignment 2 (Word Representations).

Every notebook and script of Assignment 2 imports from this file, so all
representations use exactly the same corpus and the same settings:

- CLEANED_DATA_PATH: the cleaned dataset produced in Assignment 1
- COUNT_SETTINGS:    settings for BoW / TF-IDF / N-grams (Part A)
- EMBEDDING_SETTINGS: hyperparameters for Word2Vec CBOW, Skip-gram and fastText (Part B)

Train all representations again from scratch and save them to models/ (D1):
    python representations.py
"""

import os

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DATA_PATH = os.path.join(PROJECT_DIR, "data", "cleaned_imdb_reviews.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Part A: the cleaned text is already tokenized and lowercased in Assignment 1,
# so we split on spaces and keep every token exactly as it is
# (scikit-learn's default tokenizer would drop "n't" and one-letter tokens).
COUNT_SETTINGS = {
    "tokenizer": str.split,
    "token_pattern": None,
    "lowercase": False,
}

# Part B: the same hyperparameters for every embedding model.
# CBOW and Skip-gram differ ONLY in sg (0 = CBOW, 1 = Skip-gram).
EMBEDDING_SETTINGS = {
    "vector_size": 100,   # length of each word vector
    "window": 5,          # context words on each side of the centre word
    "min_count": 5,       # ignore words that appear fewer than 5 times
    "epochs": 10,         # passes over the corpus
    "workers": 4,         # CPU threads used for training
    "seed": 42,           # random seed
}

# fastText only: character n-grams of length 3 to 6, stored in 200,000 hash buckets
# (gensim's default of 2,000,000 buckets makes the saved model about 800 MB)
FASTTEXT_SETTINGS = {
    "min_n": 3,
    "max_n": 6,
    "bucket": 200000,
}

# Files written by save_representations() (D1) and read by vectorize.py (D2)
MODEL_FILES = {
    "tfidf": os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"),
    "cbow": os.path.join(MODELS_DIR, "word2vec_cbow.kv"),
    "skipgram": os.path.join(MODELS_DIR, "word2vec_skipgram.kv"),
    "fasttext": os.path.join(MODELS_DIR, "fasttext.kv"),
}


def load_cleaned_corpus():
    """Load the Assignment 1 cleaned dataset (review_id, sentiment, cleaned_text)."""
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(
            f"{CLEANED_DATA_PATH} not found. Run 'python cleaning_pipeline.py' (Assignment 1) first."
        )
    return pd.read_csv(CLEANED_DATA_PATH, keep_default_na=False)


def load_tokenized_reviews(corpus_df):
    """Return the cleaned reviews as lists of tokens (for Word2Vec / fastText)."""
    return corpus_df["cleaned_text"].str.split().tolist()


def save_representations(tfidf_vectorizer, cbow_model, skipgram_model, fasttext_model):
    """Save the TF-IDF vectorizer and the word vectors of the three embedding models (D1)."""
    import joblib

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(tfidf_vectorizer, MODEL_FILES["tfidf"])
    cbow_model.wv.save(MODEL_FILES["cbow"])
    skipgram_model.wv.save(MODEL_FILES["skipgram"])
    fasttext_model.wv.save(MODEL_FILES["fasttext"])
    return MODEL_FILES


def build_and_save_all():
    """Fit TF-IDF and train CBOW, Skip-gram and fastText on the cleaned corpus, then save them."""
    from gensim.models import FastText, Word2Vec
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus_df = load_cleaned_corpus()
    tokenized_reviews = load_tokenized_reviews(corpus_df)

    print("Fitting TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(**COUNT_SETTINGS)
    tfidf_vectorizer.fit(corpus_df["cleaned_text"])

    print("Training Word2Vec CBOW...")
    cbow_model = Word2Vec(sentences=tokenized_reviews, sg=0, **EMBEDDING_SETTINGS)
    print("Training Word2Vec Skip-gram...")
    skipgram_model = Word2Vec(sentences=tokenized_reviews, sg=1, **EMBEDDING_SETTINGS)
    print("Training fastText...")
    fasttext_model = FastText(sentences=tokenized_reviews, **EMBEDDING_SETTINGS, **FASTTEXT_SETTINGS)

    return save_representations(tfidf_vectorizer, cbow_model, skipgram_model, fasttext_model)


if __name__ == "__main__":
    saved_files = build_and_save_all()
    print("\nSaved:")
    for name, path in saved_files.items():
        print(f"  {name:9} {path}  ({os.path.getsize(path) / 1024**2:.1f} MB)")
