import requests
import json
from datetime import datetime, timedelta
from textblob import TextBlob
import os
from dotenv import load_dotenv

load_dotenv()

class NewsAnalyzer:
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.forex_keywords = [
            'forex', 'currency', 'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD',
            'federal reserve', 'ECB', 'BOE', 'BOJ', 'central bank', 'interest rate',
            'inflation', 'GDP', 'employment', 'trade war', 'economic data'
        ]
        self.major_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD'
        ]
        
    def get_forex_sentiment(self):
        try:
            news_data = self.fetch_forex_news()
            if not news_data:
                return self.get_default_sentiment()
            
            sentiment_analysis = self.analyze_news_sentiment(news_data)
            economic_impact = self.analyze_economic_impact(news_data)
            
            return {
                'overall_sentiment': sentiment_analysis['overall'],
                'currency_specific': sentiment_analysis['currency_specific'],
                'economic_impact': economic_impact,
                'news_volume': len(news_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting forex sentiment: {str(e)}")
            return self.get_default_sentiment()
    
    def fetch_forex_news(self):
        try:
            if not self.news_api_key:
                print("News API key not found. Using alternative news sources.")
                return self.fetch_alternative_news()
            
            # NewsAPI request
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': 'forex OR currency OR "central bank" OR "interest rate"',
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': (datetime.now() - timedelta(hours=24)).isoformat(),
                'pageSize': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get('articles', [])
            
        except Exception as e:
            print(f"Error fetching news from NewsAPI: {str(e)}")
            return self.fetch_alternative_news()
    
    def fetch_alternative_news(self):
        try:
            # Alternative: scrape financial news sites
            news_sources = [
                'https://www.forexfactory.com/news',
                'https://www.investing.com/news/forex-news',
                'https://www.dailyfx.com/news'
            ]
            
            articles = []
            for source in news_sources:
                try:
                    articles.extend(self.scrape_news_source(source))
                except:
                    continue
            
            return articles
            
        except Exception as e:
            print(f"Error fetching alternative news: {str(e)}")
            return []
    
    def scrape_news_source(self, url):
        try:
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            # This is a simplified scraper - you'd need to customize for each site
            for article in soup.find_all('article')[:10]:
                title_elem = article.find('h1') or article.find('h2') or article.find('h3')
                if title_elem:
                    title = title_elem.get_text().strip()
                    
                    # Check if it's forex-related
                    if any(keyword.lower() in title.lower() for keyword in self.forex_keywords):
                        articles.append({
                            'title': title,
                            'description': title,  # Simplified
                            'publishedAt': datetime.now().isoformat(),
                            'source': {'name': url}
                        })
            
            return articles
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return []
    
    def analyze_news_sentiment(self, news_data):
        try:
            sentiments = []
            currency_sentiments = {pair: [] for pair in self.major_pairs}
            
            for article in news_data:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                # Overall sentiment
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
                sentiments.append(sentiment_score)
                
                # Currency-specific sentiment
                for pair in self.major_pairs:
                    if pair in text.upper() or any(curr in text.upper() for curr in pair):
                        currency_sentiments[pair].append(sentiment_score)
            
            overall_sentiment = self.calculate_sentiment_category(sentiments)
            
            currency_specific = {}
            for pair, scores in currency_sentiments.items():
                if scores:
                    currency_specific[pair] = self.calculate_sentiment_category(scores)
                else:
                    currency_specific[pair] = 'neutral'
            
            return {
                'overall': overall_sentiment,
                'currency_specific': currency_specific,
                'sentiment_scores': sentiments
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return {'overall': 'neutral', 'currency_specific': {}, 'sentiment_scores': []}
    
    def calculate_sentiment_category(self, scores):
        if not scores:
            return 'neutral'
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 0.1:
            return 'bullish'
        elif avg_score < -0.1:
            return 'bearish'
        else:
            return 'neutral'
    
    def analyze_economic_impact(self, news_data):
        try:
            high_impact_keywords = [
                'interest rate', 'federal reserve', 'ECB', 'BOE', 'BOJ',
                'inflation', 'GDP', 'employment', 'unemployment', 'NFP',
                'central bank', 'monetary policy', 'trade war', 'brexit'
            ]
            
            impact_scores = []
            for article in news_data:
                text = f"{article.get('title', '')} {article.get('description', '')}".lower()
                
                score = 0
                for keyword in high_impact_keywords:
                    if keyword in text:
                        score += 1
                
                impact_scores.append(score)
            
            if not impact_scores:
                return 'low'
            
            avg_impact = sum(impact_scores) / len(impact_scores)
            
            if avg_impact > 2:
                return 'high'
            elif avg_impact > 1:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            print(f"Error analyzing economic impact: {str(e)}")
            return 'low'
    
    def get_economic_calendar(self):
        try:
            # Simplified economic calendar - in practice, you'd use a dedicated API
            calendar_events = [
                {
                    'time': '08:30',
                    'event': 'US Non-Farm Payrolls',
                    'impact': 'high',
                    'forecast': 'TBD',
                    'previous': 'TBD'
                },
                {
                    'time': '10:00',
                    'event': 'Eurozone GDP',
                    'impact': 'medium',
                    'forecast': 'TBD',
                    'previous': 'TBD'
                }
            ]
            
            return calendar_events
            
        except Exception as e:
            print(f"Error getting economic calendar: {str(e)}")
            return []
    
    def get_default_sentiment(self):
        return {
            'overall_sentiment': 'neutral',
            'currency_specific': {pair: 'neutral' for pair in self.major_pairs},
            'economic_impact': 'low',
            'news_volume': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_central_bank_sentiment(self, news_data):
        try:
            central_banks = {
                'FED': ['federal reserve', 'fed', 'jerome powell', 'fomc'],
                'ECB': ['european central bank', 'ecb', 'christine lagarde'],
                'BOE': ['bank of england', 'boe', 'andrew bailey'],
                'BOJ': ['bank of japan', 'boj', 'haruhiko kuroda']
            }
            
            bank_sentiments = {}
            
            for bank, keywords in central_banks.items():
                relevant_articles = []
                for article in news_data:
                    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
                    if any(keyword in text for keyword in keywords):
                        relevant_articles.append(article)
                
                if relevant_articles:
                    sentiments = []
                    for article in relevant_articles:
                        text = f"{article.get('title', '')} {article.get('description', '')}"
                        blob = TextBlob(text)
                        sentiments.append(blob.sentiment.polarity)
                    
                    bank_sentiments[bank] = self.calculate_sentiment_category(sentiments)
                else:
                    bank_sentiments[bank] = 'neutral'
            
            return bank_sentiments
            
        except Exception as e:
            print(f"Error analyzing central bank sentiment: {str(e)}")
            return {bank: 'neutral' for bank in ['FED', 'ECB', 'BOE', 'BOJ']}