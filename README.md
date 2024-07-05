# News Articles Sentiment Analysis

## Goal
Classify news articles by sentiment to track public opinion trends using the Kaggle News Headlines Dataset.

## Dataset
- **Source:** [[Kaggle News Headlines Dataset](https://www.kaggle.com/therohk/million-headlines)](https://www.kaggle.com/datasets/siddharthtyagi/news-headlines-dataset-for-stock-sentiment-analyze)
- **Description:** Tn this dataset, we have top headlines for specific companies. Based on these headlines there are labels of values zero and one. Zero basically means that stock price will have a negative impact and One means that stock price will have a popositive impact.
Top1, Top2…. these are our news headlines.

## Steps

### 1. Load and Inspect the Dataset
```python
import pandas as pd

# Load the dataset
data = pd.read_csv('path/to/dataset/abcnews-date-text.csv', encoding='ISO-8859-1')

# Inspect the dataset
print(data.head())
print(data.columns)
# Concatenate headline columns
headline_columns = ['Top' + str(i) for i in range(1, 26)]
data['all_headlines'] = data[headline_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Inspect the concatenated column
print(data[['Date', 'Label', 'all_headlines']].head())
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize
    words = word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word.lower() not in stop_words]
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Preprocess the concatenated headlines
data['processed_text'] = data['all_headlines'].apply(preprocess_text)

# Inspect the processed text
print(data[['Date', 'Label', 'processed_text']].head())

from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sid.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

data['sentiment'] = data['processed_text'].apply(get_sentiment)

# Inspect the sentiment labels
print(data[['Date', 'Label', 'processed_text', 'sentiment']].head())

import matplotlib.pyplot as plt

# Plot sentiment distribution
sentiment_counts = data['sentiment'].value_counts()
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar')
plt.title('Sentiment Distribution in News Headlines')
plt.xlabel('Sentiment')
plt.ylabel('Number of Headlines')
plt.show()
import matplotlib.pyplot as plt

# Plot sentiment distribution
sentiment_counts = data['sentiment'].value_counts()
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar')
plt.title('Sentiment Distribution in News Headlines')
plt.xlabel('Sentiment')
plt.ylabel('Number of Headlines')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Split the data into training and test sets
X = data['processed_text']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

