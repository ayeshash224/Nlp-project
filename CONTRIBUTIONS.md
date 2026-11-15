# Contribution Statement

## Assignment 2 (Phase 2) – Word Representations


### Ayesha Shahzad (Member 1) — Roll No:L1F23BSCS0220

Worked on Part A (Count-based representations) in Member_1 L1F23BSCS0220.ipynb.
Built the Bag of Words count matrix on the cleaned IMDb reviews and calculated its shape and sparsity.
Generated the TF-IDF matrix and picked a sample review to explain why rare domain words get high weights while common words drop near zero.
Tested unigrams, bigrams, and trigrams on a 10,000-review sample to check vocabulary growth and how min_df helps control sparsity.
Solved the hand-worked TF-IDF and bigram example on paper, verified the exact numbers against scikit-learn, and attached the scanned copy.
Wrote the Part A section of the final report.

### Aleena Resham (Member 2) — Roll No:L1F23BSCS0581

Worked on Part B (Learned embeddings) in Member_2 L1F23BSCS0581.ipynb.
Trained the Word2Vec CBOW model (sg=0) using the shared corpus and fixed hyperparameters.
Trained the Word2Vec Skip-gram model (sg=1) with the exact same settings so both architectures could be compared fairly.
Explained the difference between CBOW (context-to-centre) and Skip-gram (centre-to-context) training objectives and which fits our dataset better.
Trained the fastText model with character n-grams (3–6) and showed how it handles misspelled and out-of-vocabulary words where Word2Vec throws a KeyError.
Wrote the Part B section of the final report.

### Alber Khan Jadoon (Member 3) — Roll No:L1F23BSCS0186

Worked on Parts C & D (Probing, comparison, and packaging) in Member_3 L1F23BSCS0581.ipynb.
Checked top-5 nearest neighbors for domain words, measured cosine similarities between similar vs. unrelated word pairs, and tested the actor - man + woman analogy across all three embedding models.
Compared count-based methods against learned embeddings in terms of dimensions, sparsity, OOV handling, and meaning capture.
Wrote representations.py to keep settings uniform and save all trained models/vectorizers to disk (D1), and vectorize.py (vectorize_document()) so Phase 3 can load them directly without retraining (D2).



## Assignment 1 (Phase 1) – Data Acquisition, Text Processing & Cleaning

Ayesha Shahzad: HandleD dataset loading, initial exploration, Unicode normalization, HTML/noise cleaning, and regex cleanup.

Aleena Resham: Handled lowercasing, tokenization, stop-word filtering, and stemming.

Alber Khan Jadoon: Handled lemmatization, comparing stemming vs. lemmatization, building the reusable cleaning script, and saving the final cleaned dataset.
