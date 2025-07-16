import json
from datetime import datetime
from tabulate import tabulate

class ConsoleDisplay:
    def __init__(self):
        self.signals_history = []
        
    def show_signals(self, signals):
        try:
            if not signals:
                print("\n📊 No trading signals generated.")
                return
            
            print(f"\n📊 Generated {len(signals)} Trading Signals")
            print("=" * 80)
            
            # Prepare data for table
            table_data = []
            for i, signal in enumerate(signals, 1):
                table_data.append([
                    i,
                    signal['pair'],
                    signal['direction'],
                    f"{signal['strength']:.2f}",
                    f"{signal['confidence']:.2f}",
                    signal['timestamp'][:16]
                ])
            
            # Display table
            headers = ['#', 'Pair', 'Direction', 'Strength', 'Confidence', 'Time']
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
            
            print("\n📋 Signal Details:")
            print("-" * 80)
            
            for i, signal in enumerate(signals, 1):
                print(f"\n{i}. {signal['pair']} - {signal['direction']}")
                print(f"   Strength: {signal['strength']:.2f} | Confidence: {signal['confidence']:.2f}")
                print(f"   Reasoning: {signal['reasoning']}")
                
                # Show component scores
                components = signal.get('components', {})
                if components:
                    print(f"   Components:")
                    for component, score in components.items():
                        print(f"     - {component.title()}: {score:.2f}")
            
            # Save to history
            self.signals_history.extend(signals)
            
            print(f"\n💾 Signals saved to history. Total signals: {len(self.signals_history)}")
            
        except Exception as e:
            print(f"Error displaying signals: {str(e)}")
    
    def show_analysis_summary(self, chart_data, news_sentiment):
        print("\n📈 Analysis Summary")
        print("=" * 50)
        
        # Chart analysis summary
        if chart_data:
            print("\n📊 Chart Analysis:")
            
            # Show detected symbol
            if 'symbol' in chart_data:
                symbol = chart_data['symbol']
                asset_type = chart_data.get('asset_type', 'unknown')
                print(f"   📍 Detected Symbol: {symbol} ({asset_type.upper()})")
            
            if 'price_action' in chart_data:
                pa = chart_data['price_action']
                print(f"   Trend: {pa.get('trend', 'unknown').upper()}")
                print(f"   Volatility: {pa.get('volatility', 'unknown').upper()}")
                print(f"   Strength: {pa.get('strength', 0):.2f}")
            
            if 'candlestick_patterns' in chart_data:
                patterns = chart_data['candlestick_patterns']
                if patterns:
                    print(f"   Patterns found: {len(patterns)}")
                    for pattern in patterns[:3]:  # Show top 3
                        print(f"     - {pattern.get('pattern', 'unknown').title()}: {pattern.get('confidence', 0):.2f}")
            
            if 'support_resistance' in chart_data:
                levels = chart_data['support_resistance']
                support = [l for l in levels if l.get('type') == 'support']
                resistance = [l for l in levels if l.get('type') == 'resistance']
                print(f"   Support levels: {len(support)}")
                print(f"   Resistance levels: {len(resistance)}")
        
        # News sentiment summary
        if news_sentiment:
            print("\n📰 News Sentiment:")
            print(f"   Overall sentiment: {news_sentiment.get('overall_sentiment', 'unknown').upper()}")
            print(f"   Economic impact: {news_sentiment.get('economic_impact', 'unknown').upper()}")
            print(f"   News volume: {news_sentiment.get('news_volume', 0)}")
            print(f"   Data source: {news_sentiment.get('data_source', 'unknown').upper()}")
            
            # Currency-specific sentiment
            currency_sentiment = news_sentiment.get('currency_specific', {})
            if currency_sentiment:
                print("   Currency-specific sentiment:")
                for pair, sentiment in currency_sentiment.items():
                    if sentiment != 'neutral':
                        print(f"     - {pair}: {sentiment.upper()}")
    
    def export_signals_to_file(self, signals, filename=None):
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trading_signals_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(signals, f, indent=2, default=str)
            
            print(f"\n💾 Signals exported to {filename}")
            return filename
            
        except Exception as e:
            print(f"Error exporting signals: {str(e)}")
            return None
    
    def show_statistics(self):
        if not self.signals_history:
            print("\n📊 No signals in history for statistics.")
            return
        
        print(f"\n📊 Signal Statistics (Total: {len(self.signals_history)})")
        print("=" * 50)
        
        # Direction statistics
        directions = [s['direction'] for s in self.signals_history]
        buy_count = directions.count('BUY')
        sell_count = directions.count('SELL')
        hold_count = directions.count('HOLD')
        
        print(f"BUY signals: {buy_count} ({buy_count/len(directions)*100:.1f}%)")
        print(f"SELL signals: {sell_count} ({sell_count/len(directions)*100:.1f}%)")
        print(f"HOLD signals: {hold_count} ({hold_count/len(directions)*100:.1f}%)")
        
        # Confidence statistics
        confidences = [s['confidence'] for s in self.signals_history]
        avg_confidence = sum(confidences) / len(confidences)
        print(f"\nAverage confidence: {avg_confidence:.2f}")
        print(f"High confidence (>0.8): {sum(1 for c in confidences if c > 0.8)}")
        print(f"Medium confidence (0.6-0.8): {sum(1 for c in confidences if 0.6 <= c <= 0.8)}")
        print(f"Low confidence (<0.6): {sum(1 for c in confidences if c < 0.6)}")
        
        # Pair statistics
        pairs = [s['pair'] for s in self.signals_history]
        from collections import Counter
        pair_counts = Counter(pairs)
        print(f"\nMost active pairs:")
        for pair, count in pair_counts.most_common(5):
            print(f"  {pair}: {count} signals")
    
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")