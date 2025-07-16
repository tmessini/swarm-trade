import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.preprocessing import StandardScaler
import os

class ImageAnalyzer:
    def __init__(self):
        # self.scaler = StandardScaler()
        pass
        
    def analyze_chart(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return None
            
            chart_data = {
                'candlestick_patterns': self.detect_candlestick_patterns(image),
                'trend_lines': self.detect_trend_lines(image),
                'support_resistance': self.detect_support_resistance(image),
                'technical_indicators': self.extract_technical_indicators(image),
                'price_action': self.analyze_price_action(image),
                'volume_analysis': self.analyze_volume(image)
            }
            
            return chart_data
            
        except Exception as e:
            print(f"Error analyzing chart: {str(e)}")
            return None
    
    def detect_candlestick_patterns(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candlestick_data = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 1000:  # Filter for candlestick-like shapes
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    if 0.1 < aspect_ratio < 0.8:  # Candlestick-like aspect ratio
                        candlestick_data.append({
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
            
            patterns = self.identify_candlestick_patterns(candlestick_data)
            return patterns
            
        except Exception as e:
            print(f"Error detecting candlestick patterns: {str(e)}")
            return []
    
    def identify_candlestick_patterns(self, candlestick_data):
        patterns = []
        
        if len(candlestick_data) < 3:
            return patterns
        
        for i in range(2, len(candlestick_data)):
            current = candlestick_data[i]
            prev1 = candlestick_data[i-1]
            prev2 = candlestick_data[i-2]
            
            # Doji pattern detection
            if current['aspect_ratio'] > 0.3 and current['height'] < 20:
                patterns.append({
                    'pattern': 'doji',
                    'position': (current['x'], current['y']),
                    'confidence': 0.7
                })
            
            # Hammer pattern detection
            if (current['height'] > prev1['height'] * 1.5 and 
                current['aspect_ratio'] < 0.3):
                patterns.append({
                    'pattern': 'hammer',
                    'position': (current['x'], current['y']),
                    'confidence': 0.6
                })
            
            # Engulfing pattern detection
            if (current['height'] > prev1['height'] * 1.2 and
                abs(current['x'] - prev1['x']) < 10):
                patterns.append({
                    'pattern': 'engulfing',
                    'position': (current['x'], current['y']),
                    'confidence': 0.5
                })
        
        return patterns
    
    def detect_trend_lines(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            trend_lines = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate slope and angle
                    if x2 - x1 != 0:
                        slope = (y2 - y1) / (x2 - x1)
                        angle = np.arctan(slope) * 180 / np.pi
                        
                        # Filter for trend lines (not too steep)
                        if -45 < angle < 45:
                            trend_lines.append({
                                'start': (x1, y1),
                                'end': (x2, y2),
                                'slope': slope,
                                'angle': angle,
                                'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
                            })
            
            return trend_lines
            
        except Exception as e:
            print(f"Error detecting trend lines: {str(e)}")
            return []
    
    def detect_support_resistance(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect horizontal lines for support/resistance
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            levels = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter for significant horizontal lines
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 100 and h < 10:  # Horizontal line characteristics
                        levels.append({
                            'y_level': y,
                            'start_x': x,
                            'end_x': x + w,
                            'strength': area,
                            'type': 'support' if y > image.shape[0] // 2 else 'resistance'
                        })
            
            return levels
            
        except Exception as e:
            print(f"Error detecting support/resistance: {str(e)}")
            return []
    
    def extract_technical_indicators(self, image):
        try:
            # This is a simplified approach - in practice, you'd need more sophisticated
            # computer vision techniques to extract actual indicator values
            
            indicators = {
                'rsi_signal': self.detect_rsi_signal(image),
                'macd_signal': self.detect_macd_signal(image),
                'moving_averages': self.detect_moving_averages(image),
                'bollinger_bands': self.detect_bollinger_bands(image)
            }
            
            return indicators
            
        except Exception as e:
            print(f"Error extracting technical indicators: {str(e)}")
            return {}
    
    def detect_rsi_signal(self, image):
        # Simplified RSI detection based on color patterns
        # In practice, you'd need OCR or more sophisticated image analysis
        return {'signal': 'neutral', 'confidence': 0.3}
    
    def detect_macd_signal(self, image):
        # Simplified MACD detection
        return {'signal': 'neutral', 'confidence': 0.3}
    
    def detect_moving_averages(self, image):
        # Detect moving average lines
        return {'ma_cross': 'none', 'confidence': 0.3}
    
    def detect_bollinger_bands(self, image):
        # Detect Bollinger Band signals
        return {'bb_signal': 'neutral', 'confidence': 0.3}
    
    def analyze_price_action(self, image):
        try:
            # Analyze general price movement patterns
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate image gradients to detect price direction
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Overall trend direction
            mean_grad_x = np.mean(grad_x)
            mean_grad_y = np.mean(grad_y)
            
            if mean_grad_x > 5:
                trend = 'bullish'
            elif mean_grad_x < -5:
                trend = 'bearish'
            else:
                trend = 'sideways'
            
            volatility = np.std(grad_y)
            
            return {
                'trend': trend,
                'volatility': 'high' if volatility > 20 else 'low',
                'strength': abs(mean_grad_x) / 10
            }
            
        except Exception as e:
            print(f"Error analyzing price action: {str(e)}")
            return {'trend': 'unknown', 'volatility': 'unknown', 'strength': 0}
    
    def analyze_volume(self, image):
        try:
            # Look for volume bars typically at the bottom of charts
            height, width = image.shape[:2]
            bottom_section = image[int(height * 0.8):, :]
            
            gray_bottom = cv2.cvtColor(bottom_section, cv2.COLOR_BGR2GRAY)
            
            # Detect vertical bars (volume bars)
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
            detected_bars = cv2.morphologyEx(gray_bottom, cv2.MORPH_OPEN, vertical_kernel)
            
            contours, _ = cv2.findContours(detected_bars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            volume_bars = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 20:
                    x, y, w, h = cv2.boundingRect(contour)
                    volume_bars.append(h)  # Height represents volume
            
            if volume_bars:
                avg_volume = np.mean(volume_bars)
                recent_volume = np.mean(volume_bars[-5:]) if len(volume_bars) >= 5 else avg_volume
                
                volume_trend = 'increasing' if recent_volume > avg_volume else 'decreasing'
            else:
                volume_trend = 'unknown'
            
            return {
                'volume_trend': volume_trend,
                'volume_bars_detected': len(volume_bars),
                'relative_volume': recent_volume / avg_volume if volume_bars else 1.0
            }
            
        except Exception as e:
            print(f"Error analyzing volume: {str(e)}")
            return {'volume_trend': 'unknown', 'volume_bars_detected': 0, 'relative_volume': 1.0}
    
    def preprocess_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            
            # Enhance image quality
            enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
            
            # Reduce noise
            denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            return denoised
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return cv2.imread(image_path)