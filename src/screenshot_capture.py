import cv2
import numpy as np
import pyautogui
import time
from datetime import datetime
import os
from PIL import Image, ImageGrab

class ScreenshotCapture:
    def __init__(self, output_dir="screenshots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
    
    def capture_trading_screen(self, region=None):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_screenshot_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
            
            screenshot.save(filepath)
            return filepath
            
        except Exception as e:
            print(f"Error capturing screenshot: {str(e)}")
            return None
    
    def capture_specific_window(self, window_title):
        try:
            import win32gui
            import win32ui
            import win32con
            
            hwnd = win32gui.FindWindow(None, window_title)
            if hwnd:
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                width = right - left
                height = bottom - top
                
                hwndDC = win32gui.GetWindowDC(hwnd)
                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()
                
                saveBitMap = win32ui.CreateBitmap()
                saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
                
                saveDC.SelectObject(saveBitMap)
                saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
                
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                
                img = Image.frombuffer(
                    'RGB',
                    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                    bmpstr, 'raw', 'BGRX', 0, 1
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"window_screenshot_{timestamp}.png"
                filepath = os.path.join(self.output_dir, filename)
                
                img.save(filepath)
                
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwndDC)
                
                return filepath
            
        except ImportError:
            print("Windows-specific window capture not available. Using general screenshot.")
            return self.capture_trading_screen()
        except Exception as e:
            print(f"Error capturing window: {str(e)}")
            return None
    
    def capture_chart_area(self, chart_coordinates):
        try:
            x, y, width, height = chart_coordinates
            region = (x, y, width, height)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_area_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            screenshot = pyautogui.screenshot(region=region)
            screenshot.save(filepath)
            
            return filepath
            
        except Exception as e:
            print(f"Error capturing chart area: {str(e)}")
            return None
    
    def find_trading_window(self):
        common_trading_apps = [
            "MetaTrader 4",
            "MetaTrader 5", 
            "TradingView",
            "ThinkOrSwim",
            "Interactive Brokers",
            "NinjaTrader",
            "cTrader"
        ]
        
        try:
            import psutil
            
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    proc_name = proc.info['name'].lower()
                    for app in common_trading_apps:
                        if app.lower().replace(" ", "") in proc_name:
                            return app
                except:
                    continue
                    
        except ImportError:
            print("psutil not available for process detection")
            
        return None
    
    def auto_capture_trading_screen(self, interval_seconds=300):
        print(f"Starting automatic screenshot capture every {interval_seconds} seconds...")
        
        while True:
            try:
                filepath = self.capture_trading_screen()
                if filepath:
                    print(f"Screenshot saved: {filepath}")
                else:
                    print("Failed to capture screenshot")
                    
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("Stopping automatic capture...")
                break
            except Exception as e:
                print(f"Error in automatic capture: {str(e)}")
                time.sleep(interval_seconds)