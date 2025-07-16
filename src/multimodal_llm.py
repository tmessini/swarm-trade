import requests
import json
import base64
import os
from PIL import Image
import io
import time
import threading

class MultimodalLLM:
    def __init__(self, base_url="http://localhost:11434", model="llama3.2-vision:latest", timeout=300):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout  # Default 5 minutes
        self.chat_history = []
        
    def check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                
                if self.model not in available_models:
                    print(f"⚠️  Model '{self.model}' not found. Available models: {available_models}")
                    if available_models:
                        self.model = available_models[0]
                        print(f"🔄 Switching to: {self.model}")
                
                return True
            else:
                return False
        except requests.exceptions.RequestException:
            return False
    
    def analyze_chart_image(self, image_path, question=None, use_streaming=False):
        """Analyze a chart image using the multimodal LLM"""
        try:
            if not self.check_ollama_connection():
                return {
                    'error': 'Ollama not available',
                    'message': 'Please ensure Ollama is running with a multimodal model (e.g., llava:latest)',
                    'suggestion': 'Run: ollama run llama3.2-vision:latest'
                }
            
            # Convert image to base64
            image_b64 = self.image_to_base64(image_path)
            if not image_b64:
                return {'error': 'Failed to encode image'}
            
            # Default question if none provided
            if not question:
                question = self.get_default_analysis_prompt()
            
            # Prepare the request
            payload = {
                "model": self.model,
                "prompt": question,
                "images": [image_b64],
                "stream": use_streaming,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2000
                }
            }
            
            # Make the request with extended timeout and retry logic
            max_retries = 3
            timeout = self.timeout
            
            for attempt in range(max_retries):
                try:
                    print(f"🔄 Attempting LLM analysis (attempt {attempt + 1}/{max_retries})...")
                    print(f"⏱️  Timeout set to {timeout} seconds ({timeout//60} minutes)")
                    print("💡 This may take a while for complex images. Please be patient...")
                    
                    # Start progress indicator
                    progress_stop = threading.Event()
                    progress_thread = threading.Thread(target=self.show_progress, args=(progress_stop,))
                    progress_thread.daemon = True
                    progress_thread.start()
                    
                    start_time = time.time()
                    try:
                        response = requests.post(
                            f"{self.base_url}/api/generate",
                            json=payload,
                            timeout=timeout
                        )
                    finally:
                        progress_stop.set()
                        progress_thread.join(timeout=1)
                    
                    elapsed_time = time.time() - start_time
                    print(f"\n✅ Response received in {elapsed_time:.1f} seconds")
                    
                    # If successful, break out of retry loop
                    break
                    
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        print(f"⏰ Timeout on attempt {attempt + 1}, retrying...")
                        time.sleep(2)  # Wait 2 seconds before retry
                        continue
                    else:
                        return {
                            'error': 'Request timeout',
                            'message': f'LLM analysis timed out after {timeout} seconds ({timeout//60} minutes). The model might be busy or the image is complex.',
                            'suggestion': 'Try with a simpler question or wait a moment and try again.'
                        }
                except requests.exceptions.RequestException as e:
                    return {
                        'error': 'Request failed',
                        'message': str(e),
                        'suggestion': 'Check if Ollama is running: ollama serve'
                    }
            
            if response.status_code == 200:
                if use_streaming:
                    # Handle streaming response
                    analysis = ""
                    print("\n🤖 LLM Response (streaming):")
                    print("=" * 60)
                    
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if 'response' in chunk:
                                    token = chunk['response']
                                    print(token, end='', flush=True)
                                    analysis += token
                                    
                                if chunk.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    print("\n" + "=" * 60)
                else:
                    # Handle regular response
                    result = response.json()
                    analysis = result.get('response', '')
                
                # Add to chat history
                self.chat_history.append({
                    'type': 'image_analysis',
                    'question': question,
                    'response': analysis,
                    'image_path': image_path,
                    'timestamp': time.time()
                })
                
                return {
                    'analysis': analysis,
                    'model': self.model,
                    'success': True
                }
            else:
                return {
                    'error': 'API request failed',
                    'status_code': response.status_code,
                    'message': response.text
                }
                
        except Exception as e:
            return {
                'error': 'Exception occurred',
                'message': str(e)
            }
    
    def ask_question_about_chart(self, image_path, question):
        """Ask a specific question about the chart"""
        return self.analyze_chart_image(image_path, question)
    
    def quick_analysis(self, image_path):
        """Get a quick analysis with shorter prompt for faster response"""
        quick_question = """
        Quick analysis of this chart:
        1. Symbol?
        2. Trend (up/down/sideways)?
        3. Buy/Sell/Hold recommendation?
        4. Key level to watch?
        
        Keep it brief and actionable.
        """
        return self.analyze_chart_image(image_path, quick_question)
    
    def get_trading_signals_from_llm(self, image_path):
        """Get trading signals using LLM analysis"""
        question = """
        Analyze this trading chart and provide trading signals. Please consider:
        
        1. What trading symbol/instrument is shown?
        2. What is the current trend direction?
        3. Are there any chart patterns visible?
        4. What are the key support and resistance levels?
        5. What does the volume indicate?
        6. Based on your analysis, what trading signal would you give (BUY/SELL/HOLD)?
        7. What is your confidence level (1-10)?
        8. What are the key risk factors?
        
        Please provide a structured analysis with clear reasoning.
        """
        
        return self.analyze_chart_image(image_path, question)
    
    def get_chart_pattern_analysis(self, image_path):
        """Get detailed chart pattern analysis"""
        question = """
        Focus on identifying chart patterns in this trading chart:
        
        1. What candlestick patterns do you see?
        2. Are there any technical patterns (triangles, head & shoulders, flags, etc.)?
        3. What do the moving averages indicate?
        4. Are there any breakout or breakdown signals?
        5. What does the overall price action suggest?
        
        Please be specific about the patterns you identify and their implications.
        """
        
        return self.analyze_chart_image(image_path, question)
    
    def get_risk_assessment(self, image_path):
        """Get risk assessment from the chart"""
        question = """
        Analyze this trading chart from a risk management perspective:
        
        1. What is the current volatility level?
        2. Where would you place stop-loss levels?
        3. What are the potential profit targets?
        4. What is the risk-to-reward ratio?
        5. Are there any major risk factors visible?
        6. What position sizing would you recommend?
        
        Please provide practical risk management advice.
        """
        
        return self.analyze_chart_image(image_path, question)
    
    def chat_with_chart(self, image_path, user_question):
        """Have a conversational chat about the chart"""
        # Add context from previous conversations
        context = ""
        if self.chat_history:
            recent_history = self.chat_history[-3:]  # Last 3 exchanges
            context = "Previous conversation context:\n"
            for entry in recent_history:
                context += f"Q: {entry['question']}\nA: {entry['response'][:200]}...\n\n"
        
        full_question = f"{context}Current question: {user_question}"
        
        return self.analyze_chart_image(image_path, full_question)
    
    def get_market_sentiment_analysis(self, image_path):
        """Get market sentiment analysis from the chart"""
        question = """
        Analyze the market sentiment based on this chart:
        
        1. What does the price action suggest about market sentiment?
        2. Are buyers or sellers in control?
        3. What does the volume pattern indicate about sentiment?
        4. Are there any signs of market exhaustion or strength?
        5. How would you describe the overall market mood?
        6. What might be driving the current price action?
        
        Please provide insights into market psychology and sentiment.
        """
        
        return self.analyze_chart_image(image_path, question)
    
    def get_default_analysis_prompt(self):
        """Get the default analysis prompt"""
        return """
        Please analyze this trading chart and provide:
        
        1. Trading Symbol: What instrument/symbol is being shown?
        2. Current Trend: What is the overall trend direction?
        3. Key Levels: Identify important support and resistance levels
        4. Chart Patterns: Any technical patterns visible?
        5. Volume Analysis: What does the volume indicate?
        6. Trading Signal: Your recommendation (BUY/SELL/HOLD) with reasoning
        7. Risk Factors: Key risks to consider
        
        Please be specific and provide actionable insights.
        """
    
    def image_to_base64(self, image_path):
        """Convert image to base64 string"""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error converting image to base64: {str(e)}")
            return None
    
    def get_available_models(self):
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except:
            return []
    
    def switch_model(self, model_name):
        """Switch to a different model"""
        available_models = self.get_available_models()
        if model_name in available_models:
            self.model = model_name
            print(f"✅ Switched to model: {model_name}")
            return True
        else:
            print(f"❌ Model '{model_name}' not available. Available models: {available_models}")
            return False
    
    def get_chat_history(self):
        """Get the chat history"""
        return self.chat_history
    
    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []
        print("✅ Chat history cleared")
    
    def set_timeout(self, timeout_seconds):
        """Set the timeout for LLM requests"""
        self.timeout = timeout_seconds
        print(f"✅ Timeout set to {timeout_seconds} seconds ({timeout_seconds//60} minutes)")
    
    def get_timeout(self):
        """Get current timeout setting"""
        return self.timeout
    
    def show_progress(self, stop_event):
        """Show progress indicator while waiting for LLM response"""
        progress_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        while not stop_event.is_set():
            print(f'\r🤖 Analyzing chart... {progress_chars[i % len(progress_chars)]}', end='', flush=True)
            time.sleep(0.1)
            i += 1
        print('\r', end='', flush=True)  # Clear progress line
    
    def export_chat_history(self, filename=None):
        """Export chat history to file"""
        if not filename:
            filename = f"chart_chat_history_{int(time.time())}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.chat_history, f, indent=2)
            print(f"✅ Chat history exported to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error exporting chat history: {str(e)}")
            return None
    
    def summarize_analysis(self, image_path):
        """Get a comprehensive summary of the chart"""
        question = """
        Provide a comprehensive trading analysis summary of this chart:
        
        **EXECUTIVE SUMMARY:**
        - Symbol and timeframe
        - Overall market bias (bullish/bearish/neutral)
        - Key price levels
        - Primary trading opportunity
        
        **TECHNICAL ANALYSIS:**
        - Trend analysis
        - Support/resistance levels
        - Chart patterns
        - Technical indicators
        
        **TRADING RECOMMENDATION:**
        - Entry point
        - Stop loss
        - Take profit targets
        - Risk/reward ratio
        - Position sizing suggestion
        
        **RISK ASSESSMENT:**
        - Key risk factors
        - Market conditions
        - Volatility assessment
        
        Please format your response clearly with the above sections.
        """
        
        return self.analyze_chart_image(image_path, question)