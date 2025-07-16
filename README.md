# Swarm Trade - Forex Trading Signals

A comprehensive forex trading signals application that combines screenshot analysis of trading charts with news sentiment analysis to generate intelligent trading signals.

## Features

- **Screenshot Capture**: Automated screenshot capture of trading platforms
- **Image Analysis**: Computer vision analysis of trading charts including:
  - Candlestick pattern recognition
  - Trend line detection
  - Support/resistance level identification
  - Technical indicator extraction
  - Price action analysis
  - Volume analysis

- **News Sentiment Analysis**: Real-time forex news analysis including:
  - Overall market sentiment
  - Currency-specific sentiment
  - Economic impact assessment
  - Central bank sentiment analysis

- **Signal Generation**: Intelligent trading signals combining:
  - Technical analysis (40% weight)
  - News sentiment (30% weight)
  - Chart patterns (20% weight)
  - Volume analysis (10% weight)

- **User Interface**: Intuitive GUI with:
  - Real-time signal display
  - Analysis charts
  - Signal details and reasoning
  - Export functionality

## Installation

### Windows Installation (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd swarm-trade
```

2. Run the Windows installer:
```bash
install_windows.bat
```

3. Set up environment variables:
```bash
copy .env.example .env
# Edit .env with your API keys
```

4. Run the application:
```bash
python main.py
```

### Manual Installation

If you encounter Windows Long Path errors, try the lightweight version:

```bash
pip install -r requirements-lite.txt
```

For the full version with machine learning features:
```bash
pip install -r requirements.txt
```

### Troubleshooting Windows Long Path Issue

If you get the TensorFlow installation error:

1. **Enable Windows Long Path support:**
   - Open Group Policy Editor (gpedit.msc)
   - Navigate to: Computer Configuration > Administrative Templates > System > Filesystem
   - Enable "Enable Win32 long paths"

2. **Alternative installation:**
   ```bash
   pip install --user tensorflow
   pip install --user scikit-learn
   ```

3. **Use the lite version:**
   ```bash
   pip install -r requirements-lite.txt
   ```
   This version works without TensorFlow and scikit-learn.

## Usage

### Command Line

**Single Analysis:**
```bash
python main.py
```

**Continuous Monitoring:**
```bash
python main.py --continuous
```

### GUI Application

```bash
python -m src.ui_display
```

The GUI provides buttons for:
- Capture Screenshot
- Analyze Chart
- Get News Sentiment
- Generate Signals
- Export Signals

## Configuration

### Environment Variables

- `NEWS_API_KEY`: Your NewsAPI key for news sentiment analysis
- `CONFIDENCE_THRESHOLD`: Minimum confidence for signal generation (default: 0.6)
- `RISK_LEVEL`: Risk level for signal filtering (low/medium/high)
- `SCREENSHOT_INTERVAL`: Interval for automatic screenshots (seconds)

### Supported Trading Platforms

- MetaTrader 4
- MetaTrader 5
- TradingView
- ThinkOrSwim
- Interactive Brokers
- NinjaTrader
- cTrader

## Architecture

### Core Components

1. **ScreenshotCapture** (`src/screenshot_capture.py`)
   - Automated screenshot capture
   - Window-specific capture
   - Chart area detection

2. **ImageAnalyzer** (`src/image_analyzer.py`)
   - Computer vision analysis
   - Pattern recognition
   - Technical indicator extraction

3. **NewsAnalyzer** (`src/news_analyzer.py`)
   - Real-time news fetching
   - Sentiment analysis
   - Economic impact assessment

4. **SignalGenerator** (`src/signal_generator.py`)
   - Multi-factor signal generation
   - Confidence scoring
   - Risk filtering

5. **UIDisplay** (`src/ui_display.py`)
   - GUI interface
   - Real-time visualization
   - Signal management

### Signal Generation Algorithm

The system uses a weighted scoring approach:

```
Final Score = (Technical × 0.4) + (Sentiment × 0.3) + (Patterns × 0.2) + (Volume × 0.1)
```

Signals are generated when:
- Final score > 0.6 (BUY signal)
- Final score < -0.6 (SELL signal)
- Otherwise: HOLD

## API Integration

### News APIs

- **NewsAPI**: Primary news source
- **Forex Factory**: Economic calendar
- **Investing.com**: Market news
- **Custom scrapers**: Fallback news sources

### Trading Platforms

The system can integrate with trading platforms through:
- Screenshot analysis
- API connections (when available)
- Window automation

## Signal Output

Signals include:
- Currency pair
- Direction (BUY/SELL/HOLD)
- Strength (0-1)
- Confidence (0-1)
- Reasoning
- Component scores
- Timestamp

## Risk Management

- Confidence-based filtering
- Risk level adjustment
- Signal validation
- Historical performance tracking

## Development

### Adding New Features

1. **New Analysis Methods**: Extend `ImageAnalyzer` class
2. **Additional News Sources**: Add to `NewsAnalyzer`
3. **Custom Indicators**: Modify `SignalGenerator`
4. **UI Enhancements**: Update `UIDisplay`

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Disclaimer

This software is for educational and research purposes only. Trading forex carries significant risk and this tool should not be used as the sole basis for trading decisions. Always conduct your own research and consider consulting with a financial advisor."# swarm-trade" 
