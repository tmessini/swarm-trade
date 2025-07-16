import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from datetime import datetime
import json
import os

class UIDisplay:
    def __init__(self):
        self.root = None
        self.signals_history = []
        self.current_signals = []
        
    def create_main_window(self):
        self.root = tk.Tk()
        self.root.title("Forex Trading Signals - Swarm Trade")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Create panels
        self.create_control_panel(main_frame)
        self.create_signals_panel(main_frame)
        self.create_charts_panel(main_frame)
        self.create_status_panel(main_frame)
        
        return self.root
    
    def create_control_panel(self, parent):
        # Control panel
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Buttons
        ttk.Button(control_frame, text="Capture Screenshot", 
                  command=self.capture_screenshot).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Analyze Chart", 
                  command=self.analyze_chart).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Get News Sentiment", 
                  command=self.get_news_sentiment).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Generate Signals", 
                  command=self.generate_signals).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="Export Signals", 
                  command=self.export_signals).grid(row=0, column=4, padx=5)
        
        # Status indicators
        self.status_vars = {
            'screenshot': tk.StringVar(value="Ready"),
            'analysis': tk.StringVar(value="Ready"),
            'news': tk.StringVar(value="Ready"),
            'signals': tk.StringVar(value="Ready")
        }
        
        status_frame = ttk.Frame(control_frame)
        status_frame.grid(row=1, column=0, columnspan=5, pady=(10, 0))
        
        for i, (key, var) in enumerate(self.status_vars.items()):
            ttk.Label(status_frame, text=f"{key.title()}:").grid(row=0, column=i*2, sticky=tk.W)
            ttk.Label(status_frame, textvariable=var).grid(row=0, column=i*2+1, padx=(0, 20))
    
    def create_signals_panel(self, parent):
        # Signals panel
        signals_frame = ttk.LabelFrame(parent, text="Trading Signals", padding="5")
        signals_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        signals_frame.columnconfigure(0, weight=1)
        signals_frame.rowconfigure(0, weight=1)
        
        # Create treeview for signals
        columns = ('Pair', 'Direction', 'Strength', 'Confidence', 'Time')
        self.signals_tree = ttk.Treeview(signals_frame, columns=columns, show='headings')
        
        # Define column headings
        for col in columns:
            self.signals_tree.heading(col, text=col)
            self.signals_tree.column(col, width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(signals_frame, orient=tk.VERTICAL, command=self.signals_tree.yview)
        self.signals_tree.configure(yscrollcommand=scrollbar.set)
        
        # Grid layout
        self.signals_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind selection event
        self.signals_tree.bind('<<TreeviewSelect>>', self.on_signal_select)
        
        # Signal details frame
        details_frame = ttk.LabelFrame(signals_frame, text="Signal Details", padding="5")
        details_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.details_text = tk.Text(details_frame, height=4, wrap=tk.WORD)
        self.details_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        details_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        details_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        details_frame.columnconfigure(0, weight=1)
    
    def create_charts_panel(self, parent):
        # Charts panel
        charts_frame = ttk.LabelFrame(parent, text="Analysis Charts", padding="5")
        charts_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        charts_frame.columnconfigure(0, weight=1)
        charts_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.tight_layout(pad=3.0)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize charts
        self.update_charts()
    
    def create_status_panel(self, parent):
        # Status panel
        status_frame = ttk.LabelFrame(parent, text="System Status", padding="5")
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_text = tk.Text(status_frame, height=6, wrap=tk.WORD)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        status_scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        status_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        status_frame.columnconfigure(0, weight=1)
        
        # Add initial status message
        self.log_message("System initialized. Ready for analysis.")
    
    def show_signals(self, signals):
        try:
            self.current_signals = signals
            
            # Clear existing items
            for item in self.signals_tree.get_children():
                self.signals_tree.delete(item)
            
            # Add new signals
            for signal in signals:
                self.signals_tree.insert('', tk.END, values=(
                    signal['pair'],
                    signal['direction'],
                    f"{signal['strength']:.2f}",
                    f"{signal['confidence']:.2f}",
                    signal['timestamp'][:16]  # Show only date and time
                ))
            
            # Update charts
            self.update_charts()
            
            # Log message
            self.log_message(f"Generated {len(signals)} trading signals")
            
        except Exception as e:
            self.log_message(f"Error displaying signals: {str(e)}")
    
    def on_signal_select(self, event):
        try:
            selection = self.signals_tree.selection()
            if selection:
                item = self.signals_tree.item(selection[0])
                pair = item['values'][0]
                
                # Find corresponding signal
                signal = next((s for s in self.current_signals if s['pair'] == pair), None)
                
                if signal:
                    # Display signal details
                    self.details_text.delete(1.0, tk.END)
                    details = f"Pair: {signal['pair']}\n"
                    details += f"Direction: {signal['direction']}\n"
                    details += f"Strength: {signal['strength']:.2f}\n"
                    details += f"Confidence: {signal['confidence']:.2f}\n"
                    details += f"Reasoning: {signal['reasoning']}\n"
                    details += f"Components: {signal['components']}"
                    
                    self.details_text.insert(tk.END, details)
                    
        except Exception as e:
            self.log_message(f"Error displaying signal details: {str(e)}")
    
    def update_charts(self):
        try:
            # Clear previous charts
            self.ax1.clear()
            self.ax2.clear()
            
            if self.current_signals:
                # Chart 1: Signal Strength by Pair
                pairs = [s['pair'] for s in self.current_signals]
                strengths = [s['strength'] for s in self.current_signals]
                colors = ['green' if s['direction'] == 'BUY' else 'red' if s['direction'] == 'SELL' else 'gray' 
                         for s in self.current_signals]
                
                self.ax1.bar(pairs, strengths, color=colors, alpha=0.7)
                self.ax1.set_title('Signal Strength by Currency Pair')
                self.ax1.set_ylabel('Strength')
                self.ax1.tick_params(axis='x', rotation=45)
                
                # Chart 2: Confidence Distribution
                confidences = [s['confidence'] for s in self.current_signals]
                self.ax2.hist(confidences, bins=10, alpha=0.7, color='blue')
                self.ax2.set_title('Signal Confidence Distribution')
                self.ax2.set_xlabel('Confidence')
                self.ax2.set_ylabel('Frequency')
            else:
                # Show empty charts
                self.ax1.text(0.5, 0.5, 'No signals to display', 
                            transform=self.ax1.transAxes, ha='center', va='center')
                self.ax2.text(0.5, 0.5, 'No signals to display', 
                            transform=self.ax2.transAxes, ha='center', va='center')
            
            # Refresh canvas
            self.canvas.draw()
            
        except Exception as e:
            self.log_message(f"Error updating charts: {str(e)}")
    
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
    
    def capture_screenshot(self):
        try:
            self.status_vars['screenshot'].set("Capturing...")
            self.log_message("Starting screenshot capture...")
            
            from .screenshot_capture import ScreenshotCapture
            capture = ScreenshotCapture()
            filepath = capture.capture_trading_screen()
            
            if filepath:
                self.status_vars['screenshot'].set("Complete")
                self.log_message(f"Screenshot saved: {filepath}")
            else:
                self.status_vars['screenshot'].set("Failed")
                self.log_message("Screenshot capture failed")
                
        except Exception as e:
            self.status_vars['screenshot'].set("Error")
            self.log_message(f"Screenshot error: {str(e)}")
    
    def analyze_chart(self):
        try:
            self.status_vars['analysis'].set("Analyzing...")
            self.log_message("Starting chart analysis...")
            
            # This would typically use the latest screenshot
            # For now, just simulate the process
            from .image_analyzer import ImageAnalyzer
            analyzer = ImageAnalyzer()
            
            # Find latest screenshot
            screenshot_dir = "screenshots"
            if os.path.exists(screenshot_dir):
                files = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]
                if files:
                    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(screenshot_dir, x)))
                    filepath = os.path.join(screenshot_dir, latest_file)
                    
                    chart_data = analyzer.analyze_chart(filepath)
                    
                    self.status_vars['analysis'].set("Complete")
                    self.log_message("Chart analysis completed")
                    return chart_data
            
            self.status_vars['analysis'].set("No Screenshot")
            self.log_message("No screenshot found for analysis")
            return None
            
        except Exception as e:
            self.status_vars['analysis'].set("Error")
            self.log_message(f"Analysis error: {str(e)}")
            return None
    
    def get_news_sentiment(self):
        try:
            self.status_vars['news'].set("Fetching...")
            self.log_message("Fetching news sentiment...")
            
            from .news_analyzer import NewsAnalyzer
            analyzer = NewsAnalyzer()
            sentiment = analyzer.get_forex_sentiment()
            
            self.status_vars['news'].set("Complete")
            self.log_message("News sentiment analysis completed")
            return sentiment
            
        except Exception as e:
            self.status_vars['news'].set("Error")
            self.log_message(f"News analysis error: {str(e)}")
            return None
    
    def generate_signals(self):
        try:
            self.status_vars['signals'].set("Generating...")
            self.log_message("Generating trading signals...")
            
            # Get chart data and news sentiment
            chart_data = self.analyze_chart()
            news_sentiment = self.get_news_sentiment()
            
            if chart_data or news_sentiment:
                from .signal_generator import SignalGenerator
                generator = SignalGenerator()
                signals = generator.generate_signals(chart_data, news_sentiment)
                
                self.show_signals(signals)
                self.signals_history.extend(signals)
                
                self.status_vars['signals'].set("Complete")
                self.log_message(f"Generated {len(signals)} signals")
            else:
                self.status_vars['signals'].set("No Data")
                self.log_message("No data available for signal generation")
                
        except Exception as e:
            self.status_vars['signals'].set("Error")
            self.log_message(f"Signal generation error: {str(e)}")
    
    def export_signals(self):
        try:
            if not self.current_signals:
                messagebox.showwarning("No Signals", "No signals to export")
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(self.current_signals, f, indent=2, default=str)
                
                self.log_message(f"Signals exported to {filename}")
                messagebox.showinfo("Export Complete", f"Signals exported to {filename}")
                
        except Exception as e:
            self.log_message(f"Export error: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export signals: {str(e)}")
    
    def run(self):
        if not self.root:
            self.create_main_window()
        
        self.root.mainloop()

# Standalone usage
if __name__ == "__main__":
    app = UIDisplay()
    app.run()