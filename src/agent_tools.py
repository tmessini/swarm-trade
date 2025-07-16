import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import os
from textblob import TextBlob
import ta

class AgentTools:
    """Tools that the LLM agent can use for comprehensive analysis"""
    
    def __init__(self):
        self.available_tools = {
            "get_stock_data": self.get_stock_data,
            "get_financial_news": self.get_financial_news,
            "calculate_technical_indicators": self.calculate_technical_indicators,
            "get_market_sentiment": self.get_market_sentiment,
            "get_company_info": self.get_company_info,
            "get_earnings_calendar": self.get_earnings_calendar,
            "analyze_volume_profile": self.analyze_volume_profile,
            "get_sector_performance": self.get_sector_performance,
            "calculate_support_resistance": self.calculate_support_resistance,
            "get_economic_indicators": self.get_economic_indicators,
            "analyze_correlation": self.analyze_correlation,
            "get_options_data": self.get_options_data
        }
    
    def get_tool_descriptions(self):
        """Get descriptions of all available tools for the LLM"""
        return {
            "get_stock_data": "Fetch historical stock/forex/crypto price data. Parameters: symbol (str), period (str: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max), interval (str: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)",
            "get_financial_news": "Get latest financial news for a symbol. Parameters: symbol (str), limit (int: default 10)",
            "calculate_technical_indicators": "Calculate technical indicators for a symbol. Parameters: symbol (str), indicators (list: rsi,macd,bb,sma,ema,stoch,adx,cci,williams)",
            "get_market_sentiment": "Analyze market sentiment from news and social media. Parameters: symbol (str)",
            "get_company_info": "Get company information and fundamentals. Parameters: symbol (str)",
            "get_earnings_calendar": "Get upcoming earnings dates. Parameters: symbol (str)",
            "analyze_volume_profile": "Analyze volume patterns and profile. Parameters: symbol (str), period (str)",
            "get_sector_performance": "Get sector performance data. Parameters: sector (str: optional)",
            "calculate_support_resistance": "Calculate support and resistance levels. Parameters: symbol (str), period (str)",
            "get_economic_indicators": "Get economic indicators and calendar. Parameters: country (str: US,EU,GB,JP)",
            "analyze_correlation": "Analyze correlation between symbols. Parameters: symbol1 (str), symbol2 (str)",
            "get_options_data": "Get options chain data. Parameters: symbol (str)"
        }
    
    def get_stock_data(self, symbol, period="1mo", interval="1d"):
        """Fetch stock data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return {"error": f"No data found for {symbol}"}
            
            # Convert to dictionary with recent data
            recent_data = data.tail(50)  # Last 50 periods
            
            result = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "current_price": float(data['Close'].iloc[-1]),
                "previous_close": float(data['Close'].iloc[-2]),
                "volume": int(data['Volume'].iloc[-1]),
                "high_52w": float(data['High'].max()),
                "low_52w": float(data['Low'].min()),
                "price_change": float(data['Close'].iloc[-1] - data['Close'].iloc[-2]),
                "price_change_pct": float((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100),
                "avg_volume": float(data['Volume'].mean()),
                "volatility": float(data['Close'].pct_change().std() * np.sqrt(252)),
                "recent_data": {
                    "dates": [str(date.date()) for date in recent_data.index],
                    "open": recent_data['Open'].tolist(),
                    "high": recent_data['High'].tolist(),
                    "low": recent_data['Low'].tolist(),
                    "close": recent_data['Close'].tolist(),
                    "volume": recent_data['Volume'].tolist()
                }
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error fetching data for {symbol}: {str(e)}"}
    
    def get_financial_news(self, symbol, limit=10):
        """Get financial news for a symbol"""
        try:
            # Try to get news from yfinance
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return {"error": f"No news found for {symbol}"}
            
            processed_news = []
            for article in news[:limit]:
                processed_news.append({
                    "title": article.get('title', ''),
                    "publisher": article.get('publisher', ''),
                    "published": article.get('providerPublishTime', ''),
                    "summary": article.get('summary', ''),
                    "url": article.get('link', '')
                })
            
            return {
                "symbol": symbol,
                "news_count": len(processed_news),
                "news": processed_news
            }
            
        except Exception as e:
            return {"error": f"Error fetching news for {symbol}: {str(e)}"}
    
    def calculate_technical_indicators(self, symbol, indicators=None):
        """Calculate technical indicators"""
        try:
            if indicators is None:
                indicators = ['rsi', 'macd', 'bb', 'sma', 'ema']
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo", interval="1d")
            
            if data.empty:
                return {"error": f"No data found for {symbol}"}
            
            results = {"symbol": symbol, "indicators": {}}
            
            # RSI
            if 'rsi' in indicators:
                results["indicators"]["rsi"] = {
                    "current": float(ta.momentum.RSIIndicator(data['Close']).rsi().iloc[-1]),
                    "signal": "oversold" if ta.momentum.RSIIndicator(data['Close']).rsi().iloc[-1] < 30 else "overbought" if ta.momentum.RSIIndicator(data['Close']).rsi().iloc[-1] > 70 else "neutral"
                }
            
            # MACD
            if 'macd' in indicators:
                macd_indicator = ta.trend.MACD(data['Close'])
                macd_line = macd_indicator.macd().iloc[-1]
                signal_line = macd_indicator.macd_signal().iloc[-1]
                results["indicators"]["macd"] = {
                    "macd": float(macd_line),
                    "signal": float(signal_line),
                    "histogram": float(macd_indicator.macd_diff().iloc[-1]),
                    "signal_type": "bullish" if macd_line > signal_line else "bearish"
                }
            
            # Bollinger Bands
            if 'bb' in indicators:
                bb_indicator = ta.volatility.BollingerBands(data['Close'])
                current_price = data['Close'].iloc[-1]
                bb_upper = bb_indicator.bollinger_hband().iloc[-1]
                bb_lower = bb_indicator.bollinger_lband().iloc[-1]
                bb_middle = bb_indicator.bollinger_mavg().iloc[-1]
                
                results["indicators"]["bollinger_bands"] = {
                    "upper": float(bb_upper),
                    "middle": float(bb_middle),
                    "lower": float(bb_lower),
                    "position": "above_upper" if current_price > bb_upper else "below_lower" if current_price < bb_lower else "middle"
                }
            
            # Moving Averages
            if 'sma' in indicators:
                sma_20 = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator().iloc[-1]
                sma_50 = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator().iloc[-1]
                results["indicators"]["sma"] = {
                    "sma_20": float(sma_20),
                    "sma_50": float(sma_50),
                    "trend": "bullish" if sma_20 > sma_50 else "bearish"
                }
            
            if 'ema' in indicators:
                ema_12 = ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator().iloc[-1]
                ema_26 = ta.trend.EMAIndicator(data['Close'], window=26).ema_indicator().iloc[-1]
                results["indicators"]["ema"] = {
                    "ema_12": float(ema_12),
                    "ema_26": float(ema_26),
                    "trend": "bullish" if ema_12 > ema_26 else "bearish"
                }
            
            # Stochastic
            if 'stoch' in indicators:
                stoch_indicator = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
                stoch_k = stoch_indicator.stoch().iloc[-1]
                stoch_d = stoch_indicator.stoch_signal().iloc[-1]
                results["indicators"]["stochastic"] = {
                    "k": float(stoch_k),
                    "d": float(stoch_d),
                    "signal": "oversold" if stoch_k < 20 else "overbought" if stoch_k > 80 else "neutral"
                }
            
            return results
            
        except Exception as e:
            return {"error": f"Error calculating indicators for {symbol}: {str(e)}"}
    
    def get_market_sentiment(self, symbol):
        """Analyze market sentiment"""
        try:
            # Get news sentiment
            news_data = self.get_financial_news(symbol, limit=20)
            
            if "error" in news_data:
                return news_data
            
            sentiments = []
            for article in news_data["news"]:
                text = f"{article['title']} {article['summary']}"
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            # Get stock performance for momentum
            stock_data = self.get_stock_data(symbol, period="1mo")
            
            momentum_score = 0
            if "error" not in stock_data:
                momentum_score = stock_data["price_change_pct"] / 100
            
            # Combined sentiment score
            combined_sentiment = (avg_sentiment * 0.7 + momentum_score * 0.3)
            
            return {
                "symbol": symbol,
                "news_sentiment": float(avg_sentiment),
                "momentum_score": float(momentum_score),
                "combined_sentiment": float(combined_sentiment),
                "sentiment_label": "bullish" if combined_sentiment > 0.1 else "bearish" if combined_sentiment < -0.1 else "neutral",
                "confidence": min(abs(combined_sentiment) * 2, 1.0),
                "news_count": len(sentiments)
            }
            
        except Exception as e:
            return {"error": f"Error analyzing sentiment for {symbol}: {str(e)}"}
    
    def get_company_info(self, symbol):
        """Get company information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return {"error": f"No company info found for {symbol}"}
            
            result = {
                "symbol": symbol,
                "company_name": info.get('longName', ''),
                "sector": info.get('sector', ''),
                "industry": info.get('industry', ''),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "dividend_yield": info.get('dividendYield', 0),
                "beta": info.get('beta', 0),
                "52_week_high": info.get('fiftyTwoWeekHigh', 0),
                "52_week_low": info.get('fiftyTwoWeekLow', 0),
                "description": info.get('longBusinessSummary', '')[:500] + "..." if info.get('longBusinessSummary', '') else ""
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error getting company info for {symbol}: {str(e)}"}
    
    def get_earnings_calendar(self, symbol):
        """Get earnings calendar"""
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is None or calendar.empty:
                return {"error": f"No earnings calendar found for {symbol}"}
            
            result = {
                "symbol": symbol,
                "earnings_date": str(calendar.index[0]) if not calendar.empty else "N/A",
                "eps_estimate": float(calendar.iloc[0]['Earnings Estimate']) if not calendar.empty else 0
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error getting earnings calendar for {symbol}: {str(e)}"}
    
    def analyze_volume_profile(self, symbol, period="1mo"):
        """Analyze volume patterns"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1d")
            
            if data.empty:
                return {"error": f"No data found for {symbol}"}
            
            # Volume analysis
            avg_volume = data['Volume'].mean()
            recent_volume = data['Volume'].iloc[-5:].mean()  # Last 5 days
            volume_trend = "increasing" if recent_volume > avg_volume * 1.2 else "decreasing" if recent_volume < avg_volume * 0.8 else "stable"
            
            # Price-volume relationship
            price_changes = data['Close'].pct_change()
            volume_changes = data['Volume'].pct_change()
            correlation = price_changes.corr(volume_changes)
            
            return {
                "symbol": symbol,
                "avg_volume": float(avg_volume),
                "recent_volume": float(recent_volume),
                "volume_trend": volume_trend,
                "price_volume_correlation": float(correlation) if not np.isnan(correlation) else 0,
                "volume_ratio": float(recent_volume / avg_volume),
                "analysis": "healthy" if volume_trend == "increasing" and correlation > 0.3 else "concerning" if volume_trend == "decreasing" else "neutral"
            }
            
        except Exception as e:
            return {"error": f"Error analyzing volume for {symbol}: {str(e)}"}
    
    def get_sector_performance(self, sector=None):
        """Get sector performance"""
        try:
            # Major sector ETFs
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financial": "XLF",
                "Consumer Discretionary": "XLY",
                "Communication": "XLC",
                "Industrial": "XLI",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Materials": "XLB"
            }
            
            results = {}
            etfs_to_check = [sector_etfs[sector]] if sector and sector in sector_etfs else list(sector_etfs.values())
            
            for sector_name, etf_symbol in sector_etfs.items():
                if etf_symbol in etfs_to_check:
                    data = self.get_stock_data(etf_symbol, period="1mo")
                    if "error" not in data:
                        results[sector_name] = {
                            "symbol": etf_symbol,
                            "performance": data["price_change_pct"],
                            "current_price": data["current_price"]
                        }
            
            return {"sector_performance": results}
            
        except Exception as e:
            return {"error": f"Error getting sector performance: {str(e)}"}
    
    def calculate_support_resistance(self, symbol, period="3mo"):
        """Calculate support and resistance levels"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1d")
            
            if data.empty:
                return {"error": f"No data found for {symbol}"}
            
            # Simple support/resistance calculation
            highs = data['High'].values
            lows = data['Low'].values
            closes = data['Close'].values
            
            # Find pivot points
            resistance_levels = []
            support_levels = []
            
            # Look for local maxima (resistance)
            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                    resistance_levels.append(highs[i])
            
            # Look for local minima (support)
            for i in range(2, len(lows) - 2):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                    support_levels.append(lows[i])
            
            # Get the most relevant levels
            current_price = closes[-1]
            
            # Filter and sort levels
            resistance_levels = sorted([r for r in resistance_levels if r > current_price])[:3]
            support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)[:3]
            
            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "resistance_levels": [float(r) for r in resistance_levels],
                "support_levels": [float(s) for s in support_levels],
                "nearest_resistance": float(resistance_levels[0]) if resistance_levels else None,
                "nearest_support": float(support_levels[0]) if support_levels else None
            }
            
        except Exception as e:
            return {"error": f"Error calculating support/resistance for {symbol}: {str(e)}"}
    
    def get_economic_indicators(self, country="US"):
        """Get economic indicators (simplified)"""
        try:
            # This is a simplified version - in production, you'd use APIs like FRED, Alpha Vantage, etc.
            indicators = {
                "US": {
                    "gdp_growth": 2.1,
                    "inflation": 3.2,
                    "unemployment": 3.7,
                    "interest_rate": 5.25,
                    "last_updated": datetime.now().strftime("%Y-%m-%d")
                },
                "EU": {
                    "gdp_growth": 1.8,
                    "inflation": 2.9,
                    "unemployment": 6.1,
                    "interest_rate": 4.75,
                    "last_updated": datetime.now().strftime("%Y-%m-%d")
                }
            }
            
            return {
                "country": country,
                "indicators": indicators.get(country, indicators["US"])
            }
            
        except Exception as e:
            return {"error": f"Error getting economic indicators: {str(e)}"}
    
    def analyze_correlation(self, symbol1, symbol2):
        """Analyze correlation between two symbols"""
        try:
            ticker1 = yf.Ticker(symbol1)
            ticker2 = yf.Ticker(symbol2)
            
            data1 = ticker1.history(period="6mo", interval="1d")['Close']
            data2 = ticker2.history(period="6mo", interval="1d")['Close']
            
            if data1.empty or data2.empty:
                return {"error": f"No data found for {symbol1} or {symbol2}"}
            
            # Align data
            combined_data = pd.concat([data1, data2], axis=1, keys=[symbol1, symbol2]).dropna()
            
            if len(combined_data) < 30:
                return {"error": "Not enough data for correlation analysis"}
            
            # Calculate correlation
            correlation = combined_data[symbol1].corr(combined_data[symbol2])
            
            # Calculate beta (if symbol2 is a market index)
            returns1 = combined_data[symbol1].pct_change().dropna()
            returns2 = combined_data[symbol2].pct_change().dropna()
            
            beta = returns1.cov(returns2) / returns2.var() if returns2.var() != 0 else 0
            
            return {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "correlation": float(correlation),
                "beta": float(beta),
                "relationship": "strong positive" if correlation > 0.7 else "moderate positive" if correlation > 0.3 else "strong negative" if correlation < -0.7 else "moderate negative" if correlation < -0.3 else "weak",
                "data_points": len(combined_data)
            }
            
        except Exception as e:
            return {"error": f"Error analyzing correlation: {str(e)}"}
    
    def get_options_data(self, symbol):
        """Get options data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get options expiration dates
            expirations = ticker.options
            
            if not expirations:
                return {"error": f"No options data found for {symbol}"}
            
            # Get options chain for the nearest expiration
            nearest_expiration = expirations[0]
            options_chain = ticker.option_chain(nearest_expiration)
            
            calls = options_chain.calls
            puts = options_chain.puts
            
            # Get current stock price
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            # Find ATM options
            atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
            
            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "expiration": nearest_expiration,
                "atm_call": {
                    "strike": float(atm_call['strike'].iloc[0]),
                    "last_price": float(atm_call['lastPrice'].iloc[0]),
                    "volume": int(atm_call['volume'].iloc[0]) if not pd.isna(atm_call['volume'].iloc[0]) else 0,
                    "open_interest": int(atm_call['openInterest'].iloc[0]) if not pd.isna(atm_call['openInterest'].iloc[0]) else 0
                },
                "atm_put": {
                    "strike": float(atm_put['strike'].iloc[0]),
                    "last_price": float(atm_put['lastPrice'].iloc[0]),
                    "volume": int(atm_put['volume'].iloc[0]) if not pd.isna(atm_put['volume'].iloc[0]) else 0,
                    "open_interest": int(atm_put['openInterest'].iloc[0]) if not pd.isna(atm_put['openInterest'].iloc[0]) else 0
                }
            }
            
        except Exception as e:
            return {"error": f"Error getting options data for {symbol}: {str(e)}"}
    
    def execute_tool(self, tool_name, **kwargs):
        """Execute a specific tool"""
        if tool_name in self.available_tools:
            return self.available_tools[tool_name](**kwargs)
        else:
            return {"error": f"Tool '{tool_name}' not found. Available tools: {list(self.available_tools.keys())}"}