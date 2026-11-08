"""
cleaning_pipeline.py - the single, reusable cleaning pipeline for the IMDb 50K dataset.

clean_review(text) applies every cleaning step in the same order every time:
    1. Unicode normalization (NFKC)
    2. Noise removal: HTML tags, HTML entities, broken/invisible characters, extra spaces
    3. Regular expressions: URLs, emails, quotes, repeated punctuation, special symbols
    4. Case folding (lowercase)
    5. Tokenization (NLTK word_tokenize)
    6. Stop-word removal (negation words are kept)
    7. Lemmatization with POS tags (WordNet)

Running this file cleans all 50,000 reviews and saves the result to
data/cleaned_imdb_reviews.csv (columns: review_id, sentiment, cleaned_text):
    python cleaning_pipeline.py

The notebooks import their cleaning functions from this file, so every notebook
and this script always use exactly the same cleaning code.

Owner: Member 3
"""

import html
import os
import re
import time
import unicodedata

import nltk
import pandas as pd

for resource in [
    "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
]:
    nltk.download(resource, quiet=True)

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import PerceptronTagger
from nltk.tokenize import word_tokenize

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(PROJECT_DIR, "data", "cleaned_imdb_reviews.csv")


# ---------------------------------------------------------------------------
# Step 2: Unicode normalization
# ---------------------------------------------------------------------------
def normalize_unicode(text):
    # Replace the accent sign used as an apostrophe (don´t -> don't)
    text = text.replace("´", "'")
    # Replace the sign used as a double quote (¨director¨ -> "director")
    text = text.replace("¨", '"')
    # Apply NFKC Unicode normalization
    return unicodedata.normalize("NFKC", text)


# ---------------------------------------------------------------------------
# Step 3: Text cleaning (noise removal)
# ---------------------------------------------------------------------------
html_tag_pattern = r"(?i)<(?:/?(?:p|i|b|u|em|strong)|br|hr)\s*/?\s*>|<a\s+href[^<>]*>|</a\s*>"
control_char_pattern = r"[\x00-\x1f\x7f-\x9f]"
invisible_char_pattern = r"[­​-‏‪-‮⁠-⁤﻿]"

# Broken characters from old Windows encoding -> what the writer meant
broken_character_fixes = {
    "\x80": "€",     # euro sign
    "\x84": '"',     # low double quote
    "\x85": "...",   # ellipsis
    "\x8a": "Š",     # letter S with caron (in names)
    "\x8e": "Ž",     # letter Z with caron (in names)
    "\x91": "'",     # left single quote
    "\x92": "'",     # right single quote / apostrophe
    "\x93": '"',     # left double quote
    "\x94": '"',     # right double quote
    "\x96": "-",     # en dash
    "\x97": "-",     # em dash
    "\x9a": "š",     # letter s with caron (in names)
    "\x9e": "ž",     # letter z with caron (in names)
}


def remove_html_tags(text):
    return re.sub(html_tag_pattern, " ", text)


def fix_html_entities(text):
    return html.unescape(text)


def fix_unwanted_characters(text):
    for broken, fixed in broken_character_fixes.items():
        text = text.replace(broken, fixed)
    text = re.sub(control_char_pattern, " ", text)
    text = re.sub(invisible_char_pattern, "", text)
    return text


def remove_extra_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text):
    text = remove_html_tags(text)
    text = fix_html_entities(text)
    text = fix_unwanted_characters(text)
    text = remove_extra_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Step 4: Regular expressions
# ---------------------------------------------------------------------------
url_pattern = r"https?://\S+|www\.\S+"
email_pattern = r"\S+@\S+\.\S+"
odd_apostrophe_pattern = r"[`‘’]"
odd_double_quote_pattern = r"[“”]"
repeated_punctuation_pattern = r"([!?,;:\-])\1+"
long_dots_pattern = r"\.{4,}"
heart_pattern = r"<3+"
special_char_pattern = r"[^\w\s.,!?'\"():;\-/&%$]|_"


def remove_urls(text):
    return re.sub(url_pattern, " ", text)


def remove_emails(text):
    return re.sub(email_pattern, " ", text)


def standardize_quotes(text):
    text = re.sub(odd_apostrophe_pattern, "'", text)
    text = re.sub(odd_double_quote_pattern, '"', text)
    return text


def reduce_repeated_punctuation(text):
    text = re.sub(long_dots_pattern, "...", text)
    text = re.sub(repeated_punctuation_pattern, r"\1", text)
    return text


def remove_special_characters(text):
    text = re.sub(heart_pattern, " ", text)
    text = re.sub(special_char_pattern, " ", text)
    return text


def regex_process(text):
    text = remove_urls(text)
    text = remove_emails(text)
    text = standardize_quotes(text)
    text = reduce_repeated_punctuation(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Step 6: Tokenization
# ---------------------------------------------------------------------------
def tokenize_review(text):
    tokens = word_tokenize(text)
    # "fan- always" gives the token "fan-", so remove a hyphen stuck at the start or end of a token
    return [token.strip("-") if token.strip("-") else token for token in tokens]


# ---------------------------------------------------------------------------
# Step 7: Stop-word removal (negation words are kept)
# ---------------------------------------------------------------------------
english_stopwords = set(stopwords.words("english"))

negation_words = {
    "not", "no", "nor", "never", "none", "nothing", "nobody", "nowhere", "neither", "cannot", "n't",
    "ain", "aren", "couldn", "didn", "doesn", "don", "hadn", "hasn", "haven", "isn",
    "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
}

stop_words_to_remove = {
    word for word in english_stopwords
    if word not in negation_words and not word.endswith("n't")
}


def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words_to_remove]


# ---------------------------------------------------------------------------
# Step 9: Lemmatization with POS tags (the method chosen for the pipeline)
# ---------------------------------------------------------------------------
wordnet_lemmatizer = WordNetLemmatizer()
pos_tagger = PerceptronTagger()   # load the POS model only once
lemma_lookup = {}


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_tokens(tokens):
    lemmas = []
    for word, tag in pos_tagger.tag(tokens):
        pos = get_wordnet_pos(tag)
        if (word, pos) not in lemma_lookup:
            lemma_lookup[(word, pos)] = wordnet_lemmatizer.lemmatize(word, pos)
        lemmas.append(lemma_lookup[(word, pos)])
    return lemmas


# ---------------------------------------------------------------------------
# The single cleaning function
# ---------------------------------------------------------------------------
def clean_review(text):
    """Clean one raw review and return its list of cleaned tokens."""
    text = normalize_unicode(text)      # 1. Unicode normalization
    text = clean_text(text)             # 2. noise removal
    text = regex_process(text)          # 3. regular expressions
    text = text.lower()                 # 4. case folding
    tokens = tokenize_review(text)      # 5. tokenization
    tokens = remove_stopwords(tokens)   # 6. stop-word removal (negations kept)
    tokens = lemmatize_tokens(tokens)   # 7. lemmatization
    return tokens


def vocabulary_size(token_lists):
    vocabulary = set()
    for tokens in token_lists:
        vocabulary.update(tokens)
    return len(vocabulary)


def main():
    from data_loader import load_imdb_data

    print("Loading the raw dataset...")
    df = load_imdb_data()
    total = len(df)
    print("Reviews loaded:", total)

    print("\nCleaning all reviews (this takes about 15-25 minutes)...")
    start_time = time.time()
    cleaned_tokens = []
    for number, review in enumerate(df["review"], start=1):
        cleaned_tokens.append(clean_review(review))
        if number % 5000 == 0 or number == total:
            print(f"  cleaned {number}/{total} reviews ({time.time() - start_time:.0f} seconds)")

    print("\nMeasuring the vocabulary before cleaning (raw text, NLTK tokens)...")
    vocabulary_before = vocabulary_size(word_tokenize(review) for review in df["review"])
    vocabulary_after = vocabulary_size(cleaned_tokens)
    print("Vocabulary size BEFORE cleaning:", vocabulary_before)
    print("Vocabulary size AFTER cleaning: ", vocabulary_after)
    print(f"Reduction: {(1 - vocabulary_after / vocabulary_before) * 100:.1f}%")

    cleaned_df = pd.DataFrame({
        "review_id": df.index,
        "sentiment": df["sentiment"],
        "cleaned_text": [" ".join(tokens) for tokens in cleaned_tokens],
    })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cleaned_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print("\nSaved cleaned dataset to:", OUTPUT_PATH)
    print("Shape (rows, columns):", cleaned_df.shape)


if __name__ == "__main__":
    main()
