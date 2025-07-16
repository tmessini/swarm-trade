# Troubleshooting Guide

## LLM Timeout Issues

### Problem: `HTTPConnectionPool(host='localhost', port=11434): Read timed out`

This is a common issue with vision models as they require significant processing time.

### Solutions:

#### 1. **Check Ollama Status**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

#### 2. **Verify Model is Available**
```bash
# Check available models
ollama list

# If llama3.2-vision:latest is not available, install it
ollama pull llama3.2-vision:latest
```

#### 3. **Use Quick Analysis**
For faster responses, use the quick analysis option:
```bash
# In chat mode
/quick

# Or modify the code to use quick_analysis() instead of analyze_chart_image()
```

#### 4. **Increase System Resources**
- **RAM**: Vision models need at least 8GB RAM
- **CPU**: Better performance with more cores
- **GPU**: If available, Ollama will use GPU acceleration

#### 5. **Alternative Models**
If llama3.2-vision is too slow, try smaller models:
```bash
# Install smaller vision model
ollama pull llava:7b

# Or even smaller
ollama pull llava:7b-v1.6
```

Then switch in the application:
```python
# In chat mode
/switch llava:7b
```

#### 6. **Optimize Ollama Configuration**
Create/edit `~/.ollama/config.json`:
```json
{
  "num_ctx": 4096,
  "num_predict": 1024,
  "temperature": 0.7,
  "top_p": 0.9
}
```

#### 7. **Check Image Size**
Large images take longer to process:
```bash
# Check image size
ls -lh screenshots/

# If images are very large (>5MB), consider resizing them
```

#### 8. **System Monitoring**
Monitor system resources while running:
```bash
# Check CPU/Memory usage
htop

# Check GPU usage (if available)
nvidia-smi
```

## Other Common Issues

### Issue: "Ollama not available"
**Solution**: Make sure Ollama is running:
```bash
ollama serve
```

### Issue: "Model not found"
**Solution**: Install the required model:
```bash
ollama pull llama3.2-vision:latest
```

### Issue: "Screenshot capture failed"
**Solution**: 
- Make sure TradingView is open and visible
- Check screenshot permissions on your system
- Try running as administrator (Windows)

### Issue: "No trading signals generated"
**Solution**: 
- This is normal if the analysis doesn't find strong signals
- The system now includes fallback signal generation
- Try using the LLM analysis for more insights

## Performance Tips

1. **Use Quick Analysis**: Start with `/quick` for faster responses
2. **Smaller Images**: Resize screenshots to reduce processing time
3. **Specific Questions**: Ask targeted questions rather than general analysis
4. **Model Selection**: Use smaller models for faster responses
5. **System Resources**: Close unnecessary applications to free up RAM

## Chat Mode Commands

- `/quick` - Fast analysis (recommended for first try)
- `/signals` - Detailed trading signals
- `/patterns` - Chart pattern analysis
- `/risk` - Risk assessment
- `/sentiment` - Market sentiment
- `/summary` - Comprehensive analysis
- `/models` - Show available models
- `/switch [model]` - Switch to different model

## Getting Help

If issues persist:
1. Check the console output for specific error messages
2. Try the quick analysis option first
3. Verify your system meets the requirements (8GB+ RAM)
4. Consider using a smaller/faster model
5. Check Ollama documentation: https://ollama.ai/docs