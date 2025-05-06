import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import re
Load dataset
df = pd.read_csv('sentimentdataset.csv')

df['Timestamp'] = pd.to_datetime(df['Timestamp'])


df = df.drop(columns=['Id'])

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove special characters
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['Cleaned_Text'] = df['Text'].apply(preprocess_text)
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

df['Sentiment_Scores'] = df['Cleaned_Text'].apply(lambda x: sid.polarity_scores(x))
df['Compound'] = df['Sentiment_Scores'].apply(lambda x: x['compound'])

def classify_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Compound'].apply(classify_sentiment)
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.show()
positive_text = ' '.join(df[df['Sentiment'] == 'Positive']['Cleaned_Text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Positive Sentiments')
plt.axis('off')
plt.show()
df.set_index('Timestamp', inplace=True)
df_resampled = df.resample('D').agg({'Sentiment': lambda x: x.mode()[0]})

plt.figure(figsize=(12, 6))
df_resampled['Sentiment'].value_counts().plot(kind='line', marker='o')
plt.title('Sentiment Trend Over Time')
plt.ylabel('Frequency')
plt.show()
