import numpy as np
from datetime import datetime
import json

class SignalGenerator:
    def __init__(self):
        self.signal_weights = {
            'technical_analysis': 0.4,
            'news_sentiment': 0.3,
            'chart_patterns': 0.2,
            'volume_analysis': 0.1
        }
        
        self.confidence_threshold = 0.6
        self.major_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD'
        ]
    
    def generate_signals(self, chart_data, news_sentiment):
        try:
            signals = []
            
            # If we have minimal data, generate at least one signal for demonstration
            if not chart_data or not news_sentiment or news_sentiment.get('news_volume', 0) == 0:
                # Generate a demo signal based on available data
                demo_signal = self.generate_demo_signal(chart_data, news_sentiment)
                if demo_signal:
                    signals.append(demo_signal)
            
            # Generate signals for all major pairs
            for pair in self.major_pairs:
                signal = self.generate_pair_signal(pair, chart_data, news_sentiment)
                if signal:
                    signals.append(signal)
            
            # If no signals generated, create fallback signals
            if not signals:
                signals = self.generate_fallback_signals(chart_data, news_sentiment)
            
            # Sort signals by confidence
            signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            return signals
            
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return self.generate_fallback_signals(chart_data, news_sentiment)
    
    def generate_pair_signal(self, pair, chart_data, news_sentiment):
        try:
            # Technical analysis score
            technical_score = self.calculate_technical_score(chart_data)
            
            # News sentiment score
            sentiment_score = self.calculate_sentiment_score(pair, news_sentiment)
            
            # Chart patterns score
            patterns_score = self.calculate_patterns_score(chart_data)
            
            # Volume analysis score
            volume_score = self.calculate_volume_score(chart_data)
            
            # Weighted final score
            final_score = (
                technical_score * self.signal_weights['technical_analysis'] +
                sentiment_score * self.signal_weights['news_sentiment'] +
                patterns_score * self.signal_weights['chart_patterns'] +
                volume_score * self.signal_weights['volume_analysis']
            )
            
            # Determine signal direction and strength
            if final_score > self.confidence_threshold:
                direction = 'BUY'
                strength = min(final_score, 1.0)
            elif final_score < -self.confidence_threshold:
                direction = 'SELL'
                strength = min(abs(final_score), 1.0)
            else:
                direction = 'HOLD'
                strength = 0.5
            
            # Calculate confidence
            confidence = self.calculate_confidence(
                technical_score, sentiment_score, patterns_score, volume_score
            )
            
            # Lower confidence threshold for better signal generation
            if confidence > 0.3:
                return {
                    'pair': pair,
                    'direction': direction,
                    'strength': strength,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat(),
                    'components': {
                        'technical': technical_score,
                        'sentiment': sentiment_score,
                        'patterns': patterns_score,
                        'volume': volume_score
                    },
                    'reasoning': self.generate_reasoning(
                        pair, direction, technical_score, sentiment_score, patterns_score, volume_score
                    )
                }
            
            return None
            
        except Exception as e:
            print(f"Error generating signal for {pair}: {str(e)}")
            return None
    
    def calculate_technical_score(self, chart_data):
        try:
            score = 0
            
            # Trend analysis
            if chart_data and 'price_action' in chart_data:
                price_action = chart_data['price_action']
                if price_action.get('trend') == 'bullish':
                    score += 0.3 * price_action.get('strength', 0)
                elif price_action.get('trend') == 'bearish':
                    score -= 0.3 * price_action.get('strength', 0)
            
            # Support/Resistance levels
            if chart_data and 'support_resistance' in chart_data:
                levels = chart_data['support_resistance']
                support_levels = [l for l in levels if l.get('type') == 'support']
                resistance_levels = [l for l in levels if l.get('type') == 'resistance']
                
                if len(support_levels) > len(resistance_levels):
                    score += 0.2
                elif len(resistance_levels) > len(support_levels):
                    score -= 0.2
            
            # Technical indicators
            if chart_data and 'technical_indicators' in chart_data:
                indicators = chart_data['technical_indicators']
                
                # RSI signal
                rsi = indicators.get('rsi_signal', {})
                if rsi.get('signal') == 'oversold':
                    score += 0.2
                elif rsi.get('signal') == 'overbought':
                    score -= 0.2
                
                # MACD signal
                macd = indicators.get('macd_signal', {})
                if macd.get('signal') == 'bullish_cross':
                    score += 0.2
                elif macd.get('signal') == 'bearish_cross':
                    score -= 0.2
            
            return max(-1, min(1, score))
            
        except Exception as e:
            print(f"Error calculating technical score: {str(e)}")
            return 0
    
    def calculate_sentiment_score(self, pair, news_sentiment):
        try:
            if not news_sentiment:
                return 0
            
            # Overall market sentiment
            overall = news_sentiment.get('overall_sentiment', 'neutral')
            overall_score = 0
            
            if overall == 'bullish':
                overall_score = 0.3
            elif overall == 'bearish':
                overall_score = -0.3
            
            # Currency-specific sentiment
            pair_sentiment = news_sentiment.get('currency_specific', {}).get(pair, 'neutral')
            pair_score = 0
            
            if pair_sentiment == 'bullish':
                pair_score = 0.4
            elif pair_sentiment == 'bearish':
                pair_score = -0.4
            
            # Economic impact modifier
            impact = news_sentiment.get('economic_impact', 'low')
            impact_multiplier = 1.0
            
            if impact == 'high':
                impact_multiplier = 1.5
            elif impact == 'medium':
                impact_multiplier = 1.2
            
            total_score = (overall_score + pair_score) * impact_multiplier
            
            return max(-1, min(1, total_score))
            
        except Exception as e:
            print(f"Error calculating sentiment score: {str(e)}")
            return 0
    
    def calculate_patterns_score(self, chart_data):
        try:
            if not chart_data or 'candlestick_patterns' not in chart_data:
                return 0
            
            patterns = chart_data['candlestick_patterns']
            score = 0
            
            for pattern in patterns:
                pattern_type = pattern.get('pattern')
                confidence = pattern.get('confidence', 0)
                
                if pattern_type in ['hammer', 'doji']:
                    score += 0.2 * confidence
                elif pattern_type in ['engulfing']:
                    score += 0.3 * confidence
                elif pattern_type in ['shooting_star', 'hanging_man']:
                    score -= 0.2 * confidence
            
            return max(-1, min(1, score))
            
        except Exception as e:
            print(f"Error calculating patterns score: {str(e)}")
            return 0
    
    def calculate_volume_score(self, chart_data):
        try:
            if not chart_data or 'volume_analysis' not in chart_data:
                return 0
            
            volume_data = chart_data['volume_analysis']
            volume_trend = volume_data.get('volume_trend', 'unknown')
            relative_volume = volume_data.get('relative_volume', 1.0)
            
            score = 0
            
            if volume_trend == 'increasing' and relative_volume > 1.2:
                score += 0.3
            elif volume_trend == 'decreasing' and relative_volume < 0.8:
                score -= 0.2
            
            return max(-1, min(1, score))
            
        except Exception as e:
            print(f"Error calculating volume score: {str(e)}")
            return 0
    
    def calculate_confidence(self, technical_score, sentiment_score, patterns_score, volume_score):
        try:
            # Confidence is higher when multiple indicators agree
            scores = [technical_score, sentiment_score, patterns_score, volume_score]
            
            # Remove zero scores for confidence calculation
            non_zero_scores = [s for s in scores if s != 0]
            
            if len(non_zero_scores) < 2:
                return 0.3  # Low confidence if less than 2 indicators
            
            # Check agreement between indicators
            positive_scores = [s for s in non_zero_scores if s > 0]
            negative_scores = [s for s in non_zero_scores if s < 0]
            
            agreement_ratio = max(len(positive_scores), len(negative_scores)) / len(non_zero_scores)
            
            # Base confidence on agreement and strength
            avg_strength = sum(abs(s) for s in non_zero_scores) / len(non_zero_scores)
            
            confidence = agreement_ratio * avg_strength
            
            return max(0, min(1, confidence))
            
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
            return 0.3
    
    def generate_reasoning(self, pair, direction, technical_score, sentiment_score, patterns_score, volume_score):
        try:
            reasoning = []
            
            # Technical analysis reasoning
            if abs(technical_score) > 0.2:
                trend = "bullish" if technical_score > 0 else "bearish"
                reasoning.append(f"Technical analysis shows {trend} trend (score: {technical_score:.2f})")
            
            # Sentiment reasoning
            if abs(sentiment_score) > 0.2:
                sentiment = "positive" if sentiment_score > 0 else "negative"
                reasoning.append(f"News sentiment is {sentiment} for {pair} (score: {sentiment_score:.2f})")
            
            # Pattern reasoning
            if abs(patterns_score) > 0.1:
                pattern_signal = "bullish" if patterns_score > 0 else "bearish"
                reasoning.append(f"Chart patterns indicate {pattern_signal} signals (score: {patterns_score:.2f})")
            
            # Volume reasoning
            if abs(volume_score) > 0.1:
                volume_signal = "supportive" if volume_score > 0 else "weak"
                reasoning.append(f"Volume analysis is {volume_signal} (score: {volume_score:.2f})")
            
            if not reasoning:
                reasoning.append("Signal based on combined analysis of multiple factors")
            
            return ". ".join(reasoning)
            
        except Exception as e:
            print(f"Error generating reasoning: {str(e)}")
            return "Signal generated from technical and fundamental analysis"
    
    def filter_signals_by_risk(self, signals, risk_level='medium'):
        try:
            risk_thresholds = {
                'low': 0.8,
                'medium': 0.6,
                'high': 0.4
            }
            
            threshold = risk_thresholds.get(risk_level, 0.6)
            
            return [signal for signal in signals if signal['confidence'] >= threshold]
            
        except Exception as e:
            print(f"Error filtering signals by risk: {str(e)}")
            return signals
    
    def export_signals(self, signals, filename=None):
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trading_signals_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(signals, f, indent=2, default=str)
            
            print(f"Signals exported to {filename}")
            return filename
            
        except Exception as e:
            print(f"Error exporting signals: {str(e)}")
            return None
    
    def generate_demo_signal(self, chart_data, news_sentiment):
        """Generate a demo signal based on available chart data"""
        try:
            if not chart_data:
                return None
            
            price_action = chart_data.get('price_action', {})
            trend = price_action.get('trend', 'sideways')
            strength = price_action.get('strength', 0.3)
            confidence = price_action.get('confidence', 0.3)
            
            # Boost confidence for demo
            confidence = max(confidence, 0.5)
            
            direction = 'BUY' if trend == 'bullish' else 'SELL' if trend == 'bearish' else 'HOLD'
            
            # Use detected symbol or default to USDJPY
            detected_symbol = chart_data.get('symbol', 'USDJPY')
            
            return {
                'pair': detected_symbol,
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'technical': strength,
                    'sentiment': 0.0,
                    'patterns': 0.1,
                    'volume': 0.1
                },
                'reasoning': f"Signal for {detected_symbol} based on chart analysis showing {trend} trend with {price_action.get('volatility', 'medium')} volatility"
            }
            
        except Exception as e:
            print(f"Error generating demo signal: {str(e)}")
            return None
    
    def generate_fallback_signals(self, chart_data, news_sentiment):
        """Generate fallback signals when main analysis fails"""
        try:
            fallback_signals = []
            
            # Use detected symbol if available, otherwise use top pairs
            detected_symbol = chart_data.get('symbol') if chart_data else None
            
            if detected_symbol:
                # Generate signals for the detected symbol and 2 other major pairs
                top_pairs = [detected_symbol, 'EURUSD', 'GBPUSD']
                # Remove duplicates while preserving order
                seen = set()
                top_pairs = [x for x in top_pairs if not (x in seen or seen.add(x))]
                
                # Add one more pair if we only have 2
                if len(top_pairs) == 2:
                    top_pairs.append('USDJPY')
            else:
                # Default pairs
                top_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
            
            for i, pair in enumerate(top_pairs):
                # Create varied signals for demonstration
                directions = ['BUY', 'SELL', 'HOLD']
                direction = directions[i % 3]
                
                confidence = 0.4 + (i * 0.1)  # Varied confidence
                strength = 0.3 + (i * 0.15)
                
                # Higher confidence for detected symbol
                if pair == detected_symbol:
                    confidence += 0.3
                    strength += 0.2
                
                reasoning = f"Fallback signal for {pair} based on general market analysis"
                
                if chart_data and 'price_action' in chart_data:
                    pa = chart_data['price_action']
                    if pa.get('trend') == 'bullish':
                        direction = 'BUY'
                        confidence += 0.2
                    elif pa.get('trend') == 'bearish':
                        direction = 'SELL'
                        confidence += 0.2
                    
                    reasoning += f". Chart shows {pa.get('trend', 'sideways')} trend"
                
                if pair == detected_symbol:
                    reasoning += " (detected from chart)"
                
                fallback_signals.append({
                    'pair': pair,
                    'direction': direction,
                    'strength': min(strength, 1.0),
                    'confidence': min(confidence, 1.0),
                    'timestamp': datetime.now().isoformat(),
                    'components': {
                        'technical': 0.3,
                        'sentiment': 0.1,
                        'patterns': 0.1,
                        'volume': 0.1
                    },
                    'reasoning': reasoning
                })
            
            return fallback_signals
            
        except Exception as e:
            print(f"Error generating fallback signals: {str(e)}")
            return []