# Research Summary: Local Video LLM Options for Home Surveillance

## Current Setup Analysis

**Existing Implementation:**
- Model: `llava:7b-v1.6-mistral-q2_K` (3.3 GB, 2-bit quantization)
- Framework: Ollama
- Processing: Single frame every 5 seconds
- Limitation: No temporal understanding, frame-by-frame only

**Available Models in Your System:**
- `llava:7b-v1.6-mistral-q2_K` (current)
- `llava:latest` (4.7 GB, likely newer version)

## Research Findings: Local Video LLM Options

### Category 1: Ollama-Compatible Models (Easiest Migration)

#### Option A: LLaVA-NeXT-Video / LLaVA-OneVision
- **Status**: Check if available in Ollama library
- **Advantages**: 
  - Natural upgrade from LLaVA 1.6
  - Supports video, multi-image, and single-image
  - Strong transfer learning across modalities
- **Action**: 
  ```bash
  # Check if available
  ollama pull llava-next-video  # or similar name
  ollama pull llava-onevision   # alternative name
  ```
- **Resource**: https://arxiv.org/abs/2408.03326

#### Option B: PLLaVA (Parameter-free LLaVA)
- **Status**: May need Hugging Face integration
- **Advantages**:
  - Parameter-free extension of LLaVA
  - Designed for dense video captioning
  - No additional training needed
- **Action**: Check Hugging Face for availability
- **Resource**: https://arxiv.org/abs/2404.16994

### Category 2: Lightweight Video Models (< 4B parameters)

#### Option C: TinyLLaVA-Video
- **Size**: < 4 billion parameters
- **Advantages**:
  - Lightweight, suitable for local hardware
  - Modular and scalable
  - Supports multiple frame sampling methods
  - Designed for video understanding
- **Deployment**: Hugging Face + Transformers
- **Resource**: https://arxiv.org/abs/2501.15513
- **Action**: Test inference speed and quality

#### Option D: Moondream2
- **Status**: Already mentioned in your README
- **Advantages**: 
  - Very lightweight
  - Good for real-time applications
- **Action**: Check if video support exists or can be added

### Category 3: Training-Free Video Models (No Fine-tuning)

#### Option E: SlowFast-LLaVA
- **Advantages**:
  - Training-free (works with existing LLaVA)
  - Two-stream architecture:
    - Slow stream: Detailed spatial semantics
    - Fast stream: Long-range temporal context
  - Captures both local and global temporal information
  - No token budget issues
- **Deployment**: Can work with existing LLaVA models
- **Resource**: https://arxiv.org/abs/2407.15841
- **Action**: Implement two-stream processing with current LLaVA

#### Option F: LLaVA-MR (Moment Retrieval)
- **Advantages**:
  - Accurate moment retrieval
  - Contextual grounding in videos
  - Can identify specific events/timestamps
- **Use Case**: Perfect for surveillance (finding when events occurred)
- **Resource**: https://arxiv.org/abs/2411.14505
- **Action**: Evaluate for event detection use case

### Category 4: Long Video Understanding Models

#### Option G: LongVLM
- **Advantages**:
  - Efficient long video understanding
  - Hierarchical token merging
  - Local + global feature encoding
  - Handles videos of varying lengths
- **Deployment**: Hugging Face
- **Resource**: https://arxiv.org/abs/2404.03384
- **Action**: Test on long surveillance sequences

#### Option H: TimeMarker
- **Advantages**:
  - Precise temporal localization
  - Temporal Separator Tokens
  - Dynamic frame sampling
  - High-quality video dialogue
- **Resource**: https://arxiv.org/abs/2411.18211
- **Action**: Test temporal accuracy

### Category 5: Advanced Models (Higher Resources)

#### Option I: GLM-4.5V
- **Advantages**:
  - MoE architecture (efficient inference)
  - "Thinking Mode" for multi-step reasoning
  - 3D-RoPE for spatial understanding
  - Open-source
- **Challenges**: May require significant resources
- **Action**: Check if quantized versions available

## Recommended Implementation Strategy

### Phase 1: Quick Wins (Start Here)

**1. Check Ollama for Video Models**
```bash
# Try these commands
ollama pull llava-next
ollama pull llava-onevision
ollama pull llava-video
```

**2. Test `llava:latest`**
- You already have it installed (4.7 GB)
- May have better video capabilities than v1.6
- Test with frame sequences

**3. Implement Scene Change Detection**
- Add to existing code
- Trigger detailed analysis only on changes
- Reduces processing load

### Phase 2: Temporal Understanding (Medium Effort)

**Option: SlowFast-LLaVA Architecture**
- Use your existing LLaVA model
- Implement two-stream processing:
  - Process every frame at low detail (fast stream)
  - Process every Nth frame at high detail (slow stream)
- Combine features for temporal understanding
- **No new model needed** - just architecture change

### Phase 3: Model Upgrade (Higher Effort)

**Recommended: TinyLLaVA-Video**
- Lightweight (< 4B parameters)
- Designed for video
- Good balance of performance and resources
- Can run alongside or replace current model

## Spatio-Temporal Understanding Approaches

### Temporal Techniques

1. **Frame Sequence Processing**
   - Instead of single frames, process 3-5 frame windows
   - Use sliding window approach
   - Aggregate temporal features

2. **Adaptive Frame Sampling**
   - High motion → more frames
   - Static scene → fewer frames
   - Use optical flow for motion detection

3. **Scene Change Detection**
   - Histogram comparison
   - Frame difference
   - Structural similarity (SSIM)
   - Trigger detailed analysis on changes

### Spatial Techniques

1. **Object Detection Integration**
   - YOLOv8-nano (lightweight)
   - Use detections to guide LLM attention
   - Spatial relationship queries

2. **Region-of-Interest (ROI) Processing**
   - Focus LLM on important regions
   - Reduce token count
   - Improve efficiency

## Implementation Recommendations

### Immediate Next Steps (This Week)

1. **Test `llava:latest` with frame sequences**
   ```python
   # Modify your code to send multiple frames
   images = [frame1, frame2, frame3, frame4, frame5]
   response = query_the_image("Describe what happens in these frames", images)
   ```

2. **Add scene change detection**
   ```python
   def detect_scene_change(frame1, frame2, threshold=0.3):
       # Simple histogram comparison
       hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
       hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
       correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
       return correlation < threshold
   ```

3. **Check Ollama library online**
   - Visit: https://ollama.com/library
   - Search for "video" or "llava"
   - Check for newer video-capable models

### Short-term (Next 2 Weeks)

1. **Implement SlowFast-LLaVA architecture**
   - Two-stream processing with existing model
   - Test temporal understanding improvement

2. **Test TinyLLaVA-Video**
   - Download from Hugging Face
   - Compare with current LLaVA
   - Benchmark performance

### Medium-term (Next Month)

1. **Hybrid architecture**
   - Fast model for continuous monitoring
   - Detailed model for scene analysis
   - Optimize resource usage

2. **Temporal aggregation**
   - Frame sequence processing
   - Long-term context understanding
   - Event tracking over time

## Hardware Considerations

### Current Model Requirements
- `llava:7b-v1.6-mistral-q2_K`: 3.3 GB (2-bit quantization)
- `llava:latest`: 4.7 GB (likely 4-bit or 8-bit)

### Estimated Requirements for New Models
- **TinyLLaVA-Video**: ~2-4 GB (quantized)
- **SlowFast-LLaVA**: Uses existing LLaVA (no additional storage)
- **LongVLM**: ~4-8 GB (depending on quantization)
- **GLM-4.5V**: 8-16 GB+ (even quantized)

### Optimization Strategies
1. **Quantization**: Use lower precision (q2_K, q4_K)
2. **Model pruning**: Remove unnecessary parameters
3. **Token optimization**: Reduce visual tokens
4. **Efficient encoders**: FastVLM, MobileCLIP2

## Key Questions Answered

### Q: What models can I use with Ollama?
**A**: Currently LLaVA 1.6 and LLaVA latest. Check Ollama library for video-specific variants.

### Q: Do I need to change my entire setup?
**A**: No. You can:
- Test `llava:latest` with minimal code changes
- Implement SlowFast-LLaVA architecture with existing model
- Add scene change detection to current code

### Q: What's the easiest way to get temporal understanding?
**A**: 
1. Process frame sequences instead of single frames
2. Implement SlowFast-LLaVA two-stream architecture
3. Add scene change detection

### Q: What's the best model for local deployment?
**A**: For your use case:
- **Easiest**: `llava:latest` with sequence processing
- **Best balance**: TinyLLaVA-Video (if available)
- **Most efficient**: SlowFast-LLaVA architecture (no new model)

## Testing Checklist

For each option, test:
- [ ] Inference speed (FPS or latency)
- [ ] Memory usage (RAM/VRAM)
- [ ] Caption quality
- [ ] Temporal understanding (can it track events?)
- [ ] Spatial understanding (can it describe locations?)
- [ ] Resource efficiency

## Resources

### Model Repositories
- Ollama Library: https://ollama.com/library
- Hugging Face: https://huggingface.co/models
- Awesome Video LLMs: https://github.com/zyayoung/Awesome-Video-LLMs

### Key Papers
- LLaVA-OneVision: https://arxiv.org/abs/2408.03326
- SlowFast-LLaVA: https://arxiv.org/abs/2407.15841
- TinyLLaVA-Video: https://arxiv.org/abs/2501.15513
- LongVLM: https://arxiv.org/abs/2404.03384
- TimeMarker: https://arxiv.org/abs/2411.18211
- PLLaVA: https://arxiv.org/abs/2404.16994
- LLaVA-MR: https://arxiv.org/abs/2411.14505

## Conclusion

You have several viable paths forward:

1. **Minimal effort**: Test `llava:latest` with frame sequences + scene change detection
2. **Medium effort**: Implement SlowFast-LLaVA architecture with existing model
3. **Higher effort**: Integrate TinyLLaVA-Video or other Hugging Face models

The key is to start with temporal processing techniques (frame sequences, scene detection) before investing in new models. Many improvements can be achieved with architectural changes to your existing setup.

