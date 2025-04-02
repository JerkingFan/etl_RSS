import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re

# Инициализация NLTK (скачиваем все необходимые данные)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') 

def extract_keywords(text, lang='english', top_n=5):
    """Извлекаем ключевые слова из текста."""
    try:
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words(lang))
        words = [word for word in tokens if word.isalpha() and word not in stop_words]
        keyword_counts = pd.Series(words).value_counts()
        return list(keyword_counts.head(top_n).index)
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def extract_news_from_rss(rss_url):
    """Достаём новости из RSS-ленты."""
    try:
        feed = feedparser.parse(rss_url)
        news_data = []
        
        for entry in feed.entries:
            # Очистка HTML-тегов из описания
            description = BeautifulSoup(entry.description, 'html.parser').get_text() if hasattr(entry, 'description') else ""
            news_data.append({
                'title': entry.title,
                'description': description,
                'published': entry.published if hasattr(entry, 'published') else None,
                'link': entry.link,
                'keywords': extract_keywords(f"{entry.title} {description}")
            })
        
        return pd.DataFrame(news_data)
    except Exception as e:
        print(f"Error parsing RSS: {e}")
        return pd.DataFrame()

def save_to_parquet(df, path):
    """Сохраняем DataFrame в Parquet-файл."""
    try:
        df.to_parquet(path, engine='pyarrow')
        print(f"Data successfully saved to {path}")
    except Exception as e:
        print(f"Error saving to Parquet: {e}")

if __name__ == "__main__":
    # Пример RSS-ленты (BBC News)
    RSS_URL = "http://feeds.bbci.co.uk/news/rss.xml"
    
    # ETL-процесс
    print("Extracting news from RSS...")
    news_df = extract_news_from_rss(RSS_URL)
    
    if not news_df.empty:
        print("Transforming data...")
        news_df['published'] = pd.to_datetime(news_df['published'], errors='coerce')
        
        print("Saving to Parquet...")
        save_to_parquet(news_df, "news_data.parquet")
        
        print(f"Done! Data saved to news_data.parquet (total entries: {len(news_df)})")
    else:
        print("No data was extracted. Check the RSS URL or your internet connection.")
