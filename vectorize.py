"""
vectorize.py - the interface Phase 3 calls (D2).

    from vectorize import vectorize_document
    vector = vectorize_document("This movie was not good!", representation="tfidf")

representation can be "tfidf", "cbow", "skipgram" or "fasttext".
- "tfidf"    -> a sparse row vector with one value per vocabulary word
- embeddings -> the average of the word vectors of the review's tokens (100 numbers)

The raw document is cleaned with clean_review() from Assignment 1, so it is
processed exactly like the training corpus. Models are only LOADED from models/
(saved once by representations.py); nothing is retrained here.
"""

import numpy as np

from cleaning_pipeline import clean_review
from representations import MODEL_FILES

_loaded_models = {}


def load_representation(representation):
    """Load a saved representation once and keep it in memory."""
    if representation not in MODEL_FILES:
        raise ValueError(f"representation must be one of {list(MODEL_FILES)}")
    if representation not in _loaded_models:
        if representation == "tfidf":
            import joblib
            _loaded_models[representation] = joblib.load(MODEL_FILES[representation])
        else:
            from gensim.models import KeyedVectors
            _loaded_models[representation] = KeyedVectors.load(MODEL_FILES[representation])
    return _loaded_models[representation]


def vectorize_document(raw_text, representation="tfidf"):
    """Return the vector of one raw document under the chosen representation."""
    tokens = clean_review(raw_text)
    model = load_representation(representation)

    if representation == "tfidf":
        return model.transform([" ".join(tokens)])

    if representation == "fasttext":
        word_vectors = [model[token] for token in tokens]                       # fastText also builds OOV words
    else:
        word_vectors = [model[token] for token in tokens if token in model.key_to_index]

    if not word_vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(word_vectors, axis=0)
