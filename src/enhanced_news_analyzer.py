import requests
import json
from datetime import datetime, timedelta
from textblob import TextBlob
import os
from dotenv import load_dotenv
import time

load_dotenv()

class EnhancedNewsAnalyzer:
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
            # Try multiple sources for news
            news_data = self.fetch_forex_news()
            
            if not news_data:
                # Generate simulated news sentiment for demo
                return self.generate_demo_sentiment()
            
            sentiment_analysis = self.analyze_news_sentiment(news_data)
            economic_impact = self.analyze_economic_impact(news_data)
            
            return {
                'overall_sentiment': sentiment_analysis['overall'],
                'currency_specific': sentiment_analysis['currency_specific'],
                'economic_impact': economic_impact,
                'news_volume': len(news_data),
                'timestamp': datetime.now().isoformat(),
                'data_source': 'live' if news_data else 'simulated'
            }
            
        except Exception as e:
            print(f"Error getting forex sentiment: {str(e)}")
            return self.generate_demo_sentiment()
    
    def fetch_forex_news(self):
        try:
            if not self.news_api_key:
                print("No News API key found. Generating demo sentiment...")
                return self.generate_demo_news()
            
            # NewsAPI request
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': 'forex OR currency OR "central bank" OR "interest rate" OR "Federal Reserve" OR USD OR EUR',
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': (datetime.now() - timedelta(hours=24)).isoformat(),
                'pageSize': 30
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            if articles:
                print(f"✅ Fetched {len(articles)} news articles from NewsAPI")
                return articles
            else:
                print("No articles found from NewsAPI, generating demo data...")
                return self.generate_demo_news()
            
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching news: {str(e)}")
            return self.generate_demo_news()
        except Exception as e:
            print(f"Error fetching news from NewsAPI: {str(e)}")
            return self.generate_demo_news()
    
    def generate_demo_news(self):
        """Generate demo news articles for testing"""
        demo_articles = [
            {
                'title': 'Federal Reserve Signals Potential Rate Hike',
                'description': 'The Federal Reserve indicated it may raise interest rates in the coming months to combat inflation.',
                'publishedAt': datetime.now().isoformat(),
                'source': {'name': 'Demo Financial News'},
                'content': 'Federal Reserve officials suggest monetary policy tightening may be necessary.'
            },
            {
                'title': 'EUR/USD Reaches New Monthly High',
                'description': 'The Euro strengthened against the US Dollar following positive economic data from the Eurozone.',
                'publishedAt': (datetime.now() - timedelta(hours=2)).isoformat(),
                'source': {'name': 'Demo Forex News'},
                'content': 'European Central Bank remains optimistic about economic recovery.'
            },
            {
                'title': 'USD/JPY Volatility Increases on BOJ Comments',
                'description': 'Bank of Japan governor comments on monetary policy sparked increased volatility in the yen.',
                'publishedAt': (datetime.now() - timedelta(hours=4)).isoformat(),
                'source': {'name': 'Demo Market News'},
                'content': 'Japanese monetary policy remains accommodative despite global trends.'
            },
            {
                'title': 'GBP Strengthens on UK Economic Data',
                'description': 'British Pound gains strength following better-than-expected economic indicators.',
                'publishedAt': (datetime.now() - timedelta(hours=6)).isoformat(),
                'source': {'name': 'Demo Economic News'},
                'content': 'UK unemployment rate falls to lowest level in years.'
            },
            {
                'title': 'CAD Influenced by Oil Price Movements',
                'description': 'Canadian Dollar correlation with oil prices remains strong as energy markets fluctuate.',
                'publishedAt': (datetime.now() - timedelta(hours=8)).isoformat(),
                'source': {'name': 'Demo Commodity News'},
                'content': 'Oil prices surge impacts Canadian Dollar strength significantly.'
            }
        ]
        
        print(f"📰 Generated {len(demo_articles)} demo news articles")
        return demo_articles
    
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
                text_upper = text.upper()
                
                # Check for specific currency pair mentions
                for pair in self.major_pairs:
                    if pair in text_upper:
                        currency_sentiments[pair].append(sentiment_score)
                    else:
                        # Check for individual currency mentions
                        base_currency = pair[:3]
                        quote_currency = pair[3:]
                        
                        if base_currency in text_upper or quote_currency in text_upper:
                            # Apply partial weight for individual currency mentions
                            currency_sentiments[pair].append(sentiment_score * 0.7)
            
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
                'sentiment_scores': sentiments,
                'analysis_quality': 'high' if len(sentiments) > 5 else 'medium'
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return self.get_default_sentiment_analysis()
    
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
                'central bank', 'monetary policy', 'trade war', 'brexit',
                'rate hike', 'rate cut', 'quantitative easing'
            ]
            
            impact_scores = []
            high_impact_count = 0
            
            for article in news_data:
                text = f"{article.get('title', '')} {article.get('description', '')}".lower()
                
                score = 0
                for keyword in high_impact_keywords:
                    if keyword in text:
                        score += 1
                        if keyword in ['interest rate', 'federal reserve', 'ECB', 'BOE', 'BOJ']:
                            score += 1  # Double weight for central bank news
                
                impact_scores.append(score)
                if score >= 2:
                    high_impact_count += 1
            
            if not impact_scores:
                return 'low'
            
            avg_impact = sum(impact_scores) / len(impact_scores)
            
            if avg_impact > 1.5 or high_impact_count >= 2:
                return 'high'
            elif avg_impact > 0.5 or high_impact_count >= 1:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            print(f"Error analyzing economic impact: {str(e)}")
            return 'low'
    
    def generate_demo_sentiment(self):
        """Generate demo sentiment for testing purposes"""
        current_time = datetime.now()
        
        # Simulate varying sentiment based on time of day
        hour = current_time.hour
        
        if 8 <= hour <= 12:  # Morning - bullish
            overall_sentiment = 'bullish'
            base_sentiment = 0.3
        elif 13 <= hour <= 17:  # Afternoon - mixed
            overall_sentiment = 'neutral'
            base_sentiment = 0.0
        else:  # Evening/Night - bearish
            overall_sentiment = 'bearish'
            base_sentiment = -0.2
        
        # Generate currency-specific sentiment
        currency_sentiment = {}
        for pair in self.major_pairs:
            # Add some randomness based on pair
            pair_modifier = hash(pair) % 3 - 1  # -1, 0, or 1
            pair_sentiment = base_sentiment + (pair_modifier * 0.1)
            
            if pair_sentiment > 0.1:
                currency_sentiment[pair] = 'bullish'
            elif pair_sentiment < -0.1:
                currency_sentiment[pair] = 'bearish'
            else:
                currency_sentiment[pair] = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'currency_specific': currency_sentiment,
            'economic_impact': 'medium',
            'news_volume': 5,
            'timestamp': current_time.isoformat(),
            'data_source': 'demo'
        }
    
    def get_default_sentiment_analysis(self):
        return {
            'overall': 'neutral',
            'currency_specific': {pair: 'neutral' for pair in self.major_pairs},
            'sentiment_scores': [],
            'analysis_quality': 'low'
        }
    
    def get_economic_calendar(self):
        """Get upcoming economic events (demo version)"""
        try:
            today = datetime.now()
            calendar_events = [
                {
                    'time': '08:30',
                    'event': 'US Non-Farm Payrolls',
                    'impact': 'high',
                    'currency': 'USD',
                    'forecast': 'TBD',
                    'previous': 'TBD'
                },
                {
                    'time': '10:00',
                    'event': 'Eurozone CPI',
                    'impact': 'high',
                    'currency': 'EUR',
                    'forecast': 'TBD',
                    'previous': 'TBD'
                },
                {
                    'time': '14:00',
                    'event': 'FOMC Meeting Minutes',
                    'impact': 'high',
                    'currency': 'USD',
                    'forecast': 'TBD',
                    'previous': 'TBD'
                }
            ]
            
            return calendar_events
            
        except Exception as e:
            print(f"Error getting economic calendar: {str(e)}")
            return []
    
    def get_market_sentiment_summary(self):
        """Get a summary of current market sentiment"""
        try:
            sentiment_data = self.get_forex_sentiment()
            
            summary = {
                'overall_market_mood': sentiment_data['overall_sentiment'],
                'risk_sentiment': 'risk_on' if sentiment_data['overall_sentiment'] == 'bullish' else 'risk_off' if sentiment_data['overall_sentiment'] == 'bearish' else 'neutral',
                'major_drivers': self.identify_major_drivers(sentiment_data),
                'volatility_expectation': 'high' if sentiment_data['economic_impact'] == 'high' else 'normal',
                'confidence_level': 'high' if sentiment_data['news_volume'] > 10 else 'medium' if sentiment_data['news_volume'] > 5 else 'low'
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting market sentiment summary: {str(e)}")
            return {
                'overall_market_mood': 'neutral',
                'risk_sentiment': 'neutral',
                'major_drivers': ['Limited news data'],
                'volatility_expectation': 'normal',
                'confidence_level': 'low'
            }
    
    def identify_major_drivers(self, sentiment_data):
        """Identify the major market drivers based on sentiment"""
        drivers = []
        
        if sentiment_data['economic_impact'] == 'high':
            drivers.append('Central bank policy')
        
        if sentiment_data['news_volume'] > 10:
            drivers.append('High news volume')
        
        # Check for strong currency-specific sentiment
        strong_currencies = []
        for pair, sentiment in sentiment_data['currency_specific'].items():
            if sentiment != 'neutral':
                strong_currencies.append(f"{pair[:3]} {sentiment}")
        
        if strong_currencies:
            drivers.extend(strong_currencies[:3])  # Top 3
        
        if not drivers:
            drivers = ['General market sentiment']
        
        return drivers