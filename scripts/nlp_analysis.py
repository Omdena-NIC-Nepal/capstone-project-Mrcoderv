from transformers import pipeline
import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path("../data/processed")
EXTERNAL_DATA_PATH = Path("../data/external")
PROCESSED_DATA_PATH.mkdir(exist_ok=True)

def load_text_data():
    """Load text data from external sources."""
    try:
        news_df = pd.read_csv(EXTERNAL_DATA_PATH / "climate_news.csv")
        return news_df["text"].tolist()
    except FileNotFoundError:
        print("External data not found. Using sample texts.")
        return [
            "Climate change is worsening floods in Nepal",
            "New policies help farmers adapt to climate change"
        ]

def sentiment_analysis():
    """Perform sentiment analysis on climate-related texts."""
    sentiment_analyzer = pipeline("sentiment-analysis")
    
    texts = load_text_data()
    
    try:
        results = [sentiment_analyzer(text[:512])[0] for text in texts]  # Truncate to model max length
        df = pd.DataFrame({
            "text": texts,
            "sentiment": [r["label"] for r in results],
            "confidence": [r["score"] for r in results]
        })
        df.to_csv(PROCESSED_DATA_PATH / "sentiment_results.csv", index=False)
        print("Sentiment analysis results saved successfully.")
    except Exception as e:
        print(f"Error during sentiment analysis: {str(e)}")

if __name__ == "__main__":
    sentiment_analysis()