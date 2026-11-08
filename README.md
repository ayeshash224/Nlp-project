# IMDb 50K Movie Reviews – NLP Project (Phase 1)

**Assignment 1 – Data Acquisition, Text Processing & Cleaning**

## 1. Project Overview

This project focuses on acquiring, exploring, processing, and cleaning the IMDb 50K Movie Reviews dataset using Natural Language Processing (NLP) techniques.

The main objective is to transform raw movie reviews into a clean and structured dataset that can be used for further NLP tasks, including sentiment analysis.

The project is divided into three notebooks, with each team member responsible for specific steps of the assignment.

## 2. Team Members

| Member | Name | Roll Number |
|---|---|---|
| Member 1 | AYESHA SHAHZAD | L1F23BSCS0220 |
| Member 2 | ALEENA RESHAM | L1F23BSCS0581 |
| Member 3 | ALBER KHAN JADOON | L1F23BSCS0186 |

## 3. Dataset Information

**Dataset:** IMDb 50K Movie Reviews

**Source:** Kaggle

**Dataset Link:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

The dataset contains 50,000 movie reviews:

- 25,000 positive reviews
- 25,000 negative reviews

It contains two main columns:

- `review`: Movie review text.
- `sentiment`: Sentiment label (positive or negative).

The sentiment column is used as the target label for sentiment analysis.

**Note:** The original/raw dataset is not included in this GitHub repository. Each user should download it from Kaggle and place it in the appropriate project directory.

## 4. Project Structure

```text
imdb-nlp-project/
│
├── Member_1 L1F23BSCS0220.ipynb
├── Member_2 L1F23BSCS0581.ipynb
├── Member_3 L1F23BSCS0186.ipynb
│
├── data_loader.py
├── cleaning_pipeline.py
├── requirements.txt
├── README.md
├── contribution.txt
├── .gitignore
│
├── data/
│   └── cleaned_imdb_reviews.csv
│
└── report/
    └── Assignment_1_Report.pdf
```

## 5. Work Distribution

### Member 1 – AYESHA SHAHZAD

**Notebook:** `Member_1 L1F23BSCS0220.ipynb`

**Completed Steps:** 1–4

Responsibilities:

- Dataset loading and exploration.
- Calculating the number of documents and average review length.
- Analyzing target distribution.
- Exploring tokens, types, Zipf's law, and long-tail distribution.
- Identifying examples of messy text.
- Unicode normalization.
- Cleaning HTML tags, entities, and unwanted characters.
- Regular expressions for URLs, emails, ratings, and special symbols.
- Preparing `requirements.txt`.

### Member 2 – ALEENA RESHAM

**Notebook:** `Member_2 L1F23BSCS0581.ipynb`

**Completed Steps:** 5–8

Responsibilities:

- Lowercasing.
- Tokenization using Regex and NLTK.
- Comparing tokenization methods.
- Identifying tokenization challenges.
- Stopword removal while preserving important negations.
- Applying Porter stemming.
- Preparing the project `README.md`.

### Member 3 – ALBER KHAN JADOON

**Notebook:** `Member_3 L1F23BSCS0186.ipynb`

**Completed Steps:** 9–12

Responsibilities:

- Lemmatization using WordNet.
- Comparing stemming and lemmatization.
- Developing the final text-cleaning function.
- Comparing vocabulary size before and after cleaning.
- Generating the cleaned dataset.
- Preparing `cleaning_pipeline.py`.
- Preparing `data_loader.py`.
- Preparing `.gitignore`.

## 6. Text Cleaning Pipeline

The project applies NLP preprocessing techniques to prepare movie reviews for further analysis.

The preprocessing work includes:

1. Unicode normalization.
2. Removing HTML tags and unwanted text.
3. Handling URLs, emails, ratings, and special characters.
4. Lowercasing.
5. Tokenization.
6. Stopword removal while preserving important negations.
7. Stemming and lemmatization comparison.
8. Final text cleaning and vocabulary analysis.

The final pipeline should follow the implementation in `cleaning_pipeline.py`.

## 7. Requirements and Installation

**Prerequisites:**

- Python 3.10 or later.
- Jupyter Notebook.
- Git.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## 8. Running the Project

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### Step 2: Open the Project Directory

```bash
cd imdb-nlp-project
```

### Step 3: Download the Dataset

Download the IMDb 50K Movie Reviews dataset from Kaggle and place the raw dataset in the location expected by `data_loader.py`.

### Step 4: Run the Notebooks

Open Jupyter Notebook and execute the three notebooks:

- `Member_1 L1F23BSCS0220.ipynb`
- `Member_2 L1F23BSCS0581.ipynb`
- `Member_3 L1F23BSCS0186.ipynb`

### Step 5: Run Python Scripts

Run the data loader and cleaning pipeline according to their implemented commands and expected input paths.

The cleaned output is intended to be saved as:

`data/cleaned_imdb_reviews.csv`

## 9. Git Branching Strategy

The project uses three individual branches for team members and one main branch for the final integrated project.

| Branch Display Name | Actual Git Branch Name | Assigned Member |
|---|---|---|
| Branch 1 | `Branch-1` | AYESHA SHAHZAD |
| Branch 2 | `Branch-2` | ALEENA RESHAM |
| Branch 3 | `Branch-3` | ALBER KHAN JADOON |
| Main Branch | `main` | Final integrated project |

Each member works on their assigned branch and pushes their completed files to GitHub.

After completing their work, each member creates a Pull Request (PR) to merge their changes into the `main` branch.

The final integrated project is maintained in the `main` branch.

## 10. Contribution Statement

The individual contributions of all three team members are documented in:

`contribution.txt`

Each member's responsibilities and completed tasks are described in this file.

## 11. Final Report

The final assignment report is included in:

`report/Assignment_1_Report.pdf`

The report covers:

- Dataset acquisition and exploration.
- Text preprocessing and regular expressions.
- Tokenization and stopword removal.
- Stemming and lemmatization.
- Final cleaning pipeline.
- Reproducibility and team contributions.

## 12. Reproducibility

The project is organized into separate notebooks and reusable Python scripts to make the work easier to understand, execute, and reproduce.

The `requirements.txt` file lists the project dependencies, while the notebooks document the individual preprocessing steps.

The original dataset can be downloaded from Kaggle using the link provided above.

## 13. Future Work

The cleaned IMDb reviews dataset will be used in the next phase of the project for further NLP tasks, including feature extraction and sentiment classification.