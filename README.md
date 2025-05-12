# 🎬 Film Junky Union - IMDB Movie Review Sentiment Classifier

## 📌 Project Description
The Film Junky Union, a new edgy community for classic movie enthusiasts, aims to develop an intelligent system for filtering and categorizing movie reviews. This project focuses on building a sentiment analysis model capable of accurately identifying negative reviews using IMDB movie review data labeled with sentiment polarity with the minimum required threshold of 0.85. The main goal is to automate review moderation and surface only the most constructive feedback for the community. 

## 📖 Table of Contents
  - Project Description
  - Dataset
  - Tools and Libraries Used
  - Exploratory Data Analysis
  - Modeling Approach
  - Results
  - How to Install and Run
  - Conclusion
  - Credits

## 📂 Dataset
  - Source: IMDB Reviews Dataset (ACL 2011)
  - Fields:
      - `review`: Text of the review
      - `pos`: Target label (0 = negative, 1 = positive)
      - `ds_part`: Dataset split (train/test)
   
## ⚙️ Tools and Libraries Used
  - Python 3.10+
  - Pandas, NumPy
  - Matplotlib, Seaborn
  - Scikit-learn
  - SpaCy (for lemmatization)
  - LightGBM
  - NLTK
  - TF-IDF Vectorizer
  - Jupyter Notebook

## 📊 Exploratory Data Analysis
  - Confirmed class balance across training and testing datasets.
  - Visualized review lengths and sentiment distribution.
  - Observed a wide range of review lengths and vocabulary usage.

## 🧠 Modeling Approach
Four models were developed and tested:
  - Model 1: NLTK preprocessing + TF-IDF + Logistic Regression
  - Model 2: spaCy preprocessing + TF-IDF + Logistic Regression
  - Model 3: spaCy preprocessing + TF-IDF + LightGBM
  - Model 4: BERT (small subset due to CPU limitations)
Evaluation metric: F1 Score, focusing on minimizing false positives/negatives for negative reviews.

## ✅ Results
| Model | Preprocessing   | Classifier          | F1 Score | Notes                                      |
| ----- | --------------- | ------------------- | -------- | ------------------------------------------ |
| 1     | NLTK + TF-IDF   | Logistic Regression | **0.89** | High F1 but struggles with ambiguous cases |
| 2     | spaCy + TF-IDF  | Logistic Regression | 0.88     | Balanced and consistent                    |
| 3     | spaCy + TF-IDF  | LightGBM            | 0.87     | Slightly lower accuracy                    |
| 4     | BERT Embeddings | Logistic Regression | 0.91\*   | On small subset only (slow on CPU)         |
 Final Choice: Model 2 — due to consistent predictions and balance between interpretability and performance.

## 💻 How to Install and Run
  1. Clone this repository:
     ```bash
     git clone https://github.com/your-username/film-junky-sentiment.git
     cd film-junky-sentiment
  2. Install dependencies:
     ```bash
     pip install -r requirements.txt
  3. Run the notebook:
     ```bash
     jupyter notebook.ipynb

## 🧾 Conclusion
The machine learning solution classifies movie reviews into positive or negative categories. Logistic Regression with spaCy preprocessing (model 2) was chosen as the optimal model, even though Logistic Regression with NLTK + TF-IDF (model 1) acheived the highest F1 score. This was because model 2 consistently handled borderline reviews with more reliable classification which aligns better with the Film Junky Union's goal of review filtering rather than just accuracy maximization. 

## 🤝 Credits
This project was created as part of the TripleTen Data Science program. Special thanks to:
  - TripleTen instructors and peers for ongoing support and feedback.
  - Andrew L. Maas et al.: IMDB Review Dataset
  - spaCy, NLTK, scikit-learn open-source contributors
