# Sentiment Analysis using IMDB Dataset 

## Project Overview
This project involves performing **sentiment analysis** on the IMDB Dataset by applying Natural Language Processing (NLP) techniques, visualizing insights, extracting keywords using **TF-IDF** and **RAKE**, and building classification models using **SVM** and **Random Forest**. The project includes preprocessing text, extracting features, visualizing data, and evaluating models for accuracy.

---


## Dataset Information
- **Dataset**: IMDB-Dataset.csv
- **Source**: The dataset contains **50,000 movie reviews** with sentiment labels (positive/negative).
- **Columns**:
  - `review`: Textual reviews of movies.
  - `sentiment`: Label indicating if the review is positive or negative.

---

## Project Workflow
The project workflow includes:

1. **Data Preprocessing**:
   - Text cleaning (HTML tag removal, lowercasing, stopwords removal).
   - Feature extraction (word count, character count, average word length, etc.).
2. **Data Visualization**:
   - Histograms of word/character counts and stopwords.
   - Sentiment distribution.
3. **Keyword Extraction**:
   - TF-IDF and RAKE techniques for top keyword extraction.
   - Comparison using Venn diagrams and bar plots.
4. **Classification Models**:
   - Built two pipelines:
     - **Support Vector Machine (SVM)**.
     - **Random Forest Classifier**.
   - Model evaluation with confusion matrices and classification reports.
5. **Model Deployment**:
   - Saving trained models as pickle files for future use.

---


## Libraries Used
- **Data Handling**: pandas, numpy
- **Text Processing**: NLTK, re
- **Keyword Extraction**: rake_nltk, TF-IDF
- **Data Visualization**: matplotlib, seaborn, wordcloud, matplotlib_venn
- **ML Models**: SVM (LinearSVC), RandomForestClassifier
- **Evaluation**: Confusion Matrix, Classification Report

---

## Key Features
1. **Text Preprocessing**
   - Removed stopwords, HTML tags, and applied tokenization.
   - Added columns like word count, character count, stopwords count, and average word length.

2. **Keyword Extraction**
   - Extracted top 10 keywords from each review using:
     - **TF-IDF**.
     - **RAKE**.
   - Visualized comparison using **Venn Diagrams** and **Bar Plots**.

3. **Data Visualization**
   - Histograms and bar plots to show sentiment distribution, word counts, and keyword frequency.
   - Word cloud of keywords based on TF-IDF scores.

4. **Classification Models**
   - Built and trained two models using **SVM** and **Random Forest** pipelines.
   - Evaluated models using classification reports and confusion matrices.

5. **Model Testing**
   - Predictions on custom reviews like:
     - *"Wow, this is an amazing lesson"*
     - *"Wow, this sucks"*

6. **Model Saving**
   - Exported models as `.pkl` files for reusability.

---

## Results
The models achieved the following results:

### SVM Classifier:
- **Precision/Recall/F1-Score**: As shown in the **Classification Report**.
- **Confusion Matrix**: Visualized for test data.

### Random Forest Classifier:
- **Precision/Recall/F1-Score**: Competitive with SVM.
- **Confusion Matrix**: Visualized for performance comparison.

---
