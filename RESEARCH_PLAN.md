# Research Plan: Video Understanding with Local LLMs
## Educational Presentation + Practical Implementation

## Executive Summary

This document outlines a research plan that serves dual purposes:
1. **Educational Presentation**: Create a technical presentation explaining fundamental deep learning concepts (tokenization, embeddings, transformers) in the context of video understanding
2. **Practical Implementation**: Build a working LLM-based video spatio-temporal understanding model that runs on local hardware, demonstrating these concepts in practice

The project will upgrade the existing LLaVA 1.6-based home surveillance project while providing educational value through clear explanations of how tokenization, embeddings, and transformers work in video understanding systems. The implementation will be pragmatic, using quantization and parameter-efficient techniques to run on consumer hardware.

## Current State Analysis

### Existing Implementation
- **Model**: LLaVA 1.6 (7B, q2_K quantization) via Ollama
- **Architecture**: Frame-by-frame processing every 5 seconds
- **Limitations**:
  - No temporal understanding (treats frames independently)
  - Limited spatio-temporal reasoning
  - Basic captioning without scene context
  - Potential latency issues with real-time processing

### Key Requirements
- Must run on local hardware (no cloud dependencies)
- Focus on computer vision and video processing
- Spatio-temporal understanding capabilities
- Real-time or near-real-time performance
- **Educational Value**: Code and architecture should clearly demonstrate:
  - Tokenization: How video frames become tokens
  - Embeddings: How visual features are encoded as vectors
  - Transformers: How attention mechanisms process temporal sequences
- **Presentation-Ready**: Implementation should be demonstrable with clear examples

## Research Areas

### 0. Educational Foundation: Core Concepts for Presentation

#### 0.1 Tokenization in Video Understanding
**Presentation Focus**: Explain how video data is tokenized

- **Frame Tokenization**: 
  - Each video frame can be treated as a token
  - Frame patches (similar to image patches in ViT)
  - Temporal tokens (frame sequences)
  - **Action**: Implement and visualize frame tokenization process
  - **Code Example**: Show frame → patches → tokens pipeline

- **Temporal Tokenization**:
  - Video sequences as token sequences
  - Frame sampling strategies (uniform, adaptive, keyframe)
  - **Action**: Create visualizations showing different tokenization strategies
  - **Code Example**: Demonstrate uniform vs. adaptive sampling

- **Challenges and Trade-offs**:
  - Token budget limitations
  - Information loss in sampling
  - Computational cost vs. temporal resolution
  - **Action**: Document trade-offs with examples

#### 0.2 Embeddings in Video Understanding
**Presentation Focus**: Explain how tokens become meaningful vectors

- **Visual Embeddings**:
  - Vision encoder (CLIP, ViT) converts frames to embeddings
  - Spatial embeddings from frame patches
  - Temporal embeddings from frame sequences
  - **Action**: Extract and visualize embeddings (t-SNE plots)
  - **Code Example**: Show frame → embedding vector transformation

- **Embedding Space Properties**:
  - Similar frames cluster together
  - Temporal relationships in embedding space
  - Semantic similarity in vector space
  - **Action**: Create visualization showing embedding clusters
  - **Code Example**: Generate t-SNE plots of video frame embeddings

- **Multi-modal Embeddings**:
  - Visual embeddings + text embeddings alignment
  - Cross-modal attention mechanisms
  - **Action**: Demonstrate visual-text embedding alignment

#### 0.3 Transformers in Video Understanding
**Presentation Focus**: Explain how transformers process video sequences

- **Self-Attention Mechanism**:
  - How model "pays attention" to relevant frames
  - Temporal attention patterns
  - Spatial attention within frames
  - **Action**: Visualize attention weights across frames
  - **Code Example**: Extract and plot attention matrices

- **Temporal Transformer Architecture**:
  - Processing sequences of visual embeddings
  - Long-range temporal dependencies
  - Positional encoding for temporal order
  - **Action**: Implement and explain temporal transformer layer
  - **Code Example**: Show transformer processing frame sequence

- **Cross-Modal Attention**:
  - Attention between visual tokens and text tokens
  - Query-based attention (answering questions about video)
  - **Action**: Demonstrate query-video attention patterns

### 1. Video-Centric LLM Models for Local Deployment

#### 1.1 Models Available via Ollama
**Priority: HIGH** - Easiest migration path from current setup

- **LLaVA-NeXT-Video**: 
  - Natural upgrade path from LLaVA 1.6
  - Specifically trained for video tasks
  - Check availability in Ollama library
  - **Action**: Verify if available, test performance vs. current model

- **Current Ollama Video Models**:
  - Research what video-capable models are currently available in Ollama
  - Check quantization options (q2_K, q4_K, q8_0)
  - **Action**: Run `ollama list` and check Ollama library for video models

#### 1.2 Models Requiring Direct Hugging Face Integration
**Priority: MEDIUM** - More setup but potentially better performance

- **TinyLLaVA-Video**:
  - < 4B parameters (lightweight)
  - Modular and scalable
  - Supports frame sampling methods
  - **Action**: Check Hugging Face availability, test local inference
  - **Resource**: https://arxiv.org/abs/2501.15513

- **SlowFast-LLaVA**:
  - Training-free (no fine-tuning needed)
  - Two-stream design: spatial details + temporal context
  - Captures long-range temporal dependencies
  - **Action**: Evaluate architecture, test inference speed
  - **Resource**: https://arxiv.org/abs/2407.15841

- **LongVLM**:
  - Efficient long video understanding
  - Hierarchical token merging for local/global features
  - Decomposes videos into segments
  - **Action**: Test on long video sequences, evaluate memory usage
  - **Resource**: https://arxiv.org/abs/2404.03384

- **TimeMarker**:
  - Precise temporal localization
  - Temporal Separator Tokens
  - Dynamic frame sampling
  - **Action**: Test temporal accuracy, evaluate for surveillance use case
  - **Resource**: https://arxiv.org/abs/2411.18211

- **MiniGPT4-Video**:
  - Extends MiniGPT-v2 to video
  - Interleaved visual-textual token processing
  - **Action**: Check model size, test inference requirements

#### 1.3 Advanced Models (Higher Resource Requirements)
**Priority: LOW** - Evaluate based on hardware capabilities

- **GLM-4.5V**:
  - MoE architecture (better performance, lower inference cost)
  - "Thinking Mode" for multi-step reasoning
  - 3D-RoPE for spatial understanding
  - **Action**: Check if quantized versions available, test on hardware
  - **Note**: May require significant resources even with quantization

- **AuroraCap**:
  - Top-performing video captioning
  - Token merging strategy for long sequences
  - **Action**: Research availability, check if local deployment feasible

### 2. Spatio-Temporal Understanding Approaches

#### 2.1 Temporal Modeling Techniques

**Frame Sampling Strategies**:
- Uniform sampling (current approach)
- Adaptive sampling based on motion detection
- Keyframe extraction using scene change detection
- **Action**: Implement and compare different sampling methods

**Temporal Aggregation**:
- Frame-level features → temporal pooling
- Attention mechanisms across time
- Temporal convolution or transformer layers
- **Action**: Research and implement temporal feature aggregation

#### 2.2 Spatial Understanding Enhancement

**Object Detection Integration**:
- Combine LLM with YOLO/Detectron2 for object detection
- Use detections to guide LLM attention
- **Action**: Evaluate lightweight object detection models (YOLOv8-nano, etc.)

**Scene Understanding**:
- Room/area classification
- Spatial relationship understanding (left, right, near, far)
- **Action**: Test if video LLMs can handle spatial queries

#### 2.3 Hybrid Approaches

**Scene Change Detection + Captioning**:
- Detect scene changes using computer vision
- Generate detailed captions only on scene changes
- Maintain short-term captions for continuous monitoring
- **Action**: Implement scene change detector, integrate with LLM

**Multi-Model Pipeline**:
- Fast model for continuous monitoring (every frame)
- Detailed model for scene analysis (on scene changes)
- **Action**: Design and test multi-model architecture

### 3. Model Architecture for Presentation

#### 3.1 Proposed Architecture (Presentation-Ready)
**Goal**: Clear, explainable architecture demonstrating tokenization, embeddings, and transformers

```
Input: Video Frames
    ↓
[Visual Encoder] → Frame Embeddings (Tokenization + Embedding)
    ↓
[Temporal Transformer] → Temporal Context (Transformer Attention)
    ↓
[LLM Head] → Text Output (Language Generation)
```

**Components**:
1. **Visual Encoder** (Tokenization + Embedding):
   - Lightweight pre-trained vision model (CLIP, ViT, or MobileCLIP2)
   - Converts frames to patch embeddings
   - **Educational Value**: Shows frame → tokens → embeddings
   - **Action**: Implement with clear visualization of the process

2. **Temporal Transformer** (Transformer Architecture):
   - Processes sequence of visual embeddings
   - Self-attention across temporal dimension
   - Captures long-range dependencies
   - **Educational Value**: Demonstrates attention mechanism
   - **Action**: Implement with attention visualization

3. **LLM Head** (Language Generation):
   - Small, quantized language model
   - Takes temporal features + text query
   - Generates textual response
   - **Educational Value**: Shows how visual understanding → language
   - **Action**: Integrate quantized LLM (via Ollama or llama.cpp)

#### 3.2 Implementation Approach
- **Quantization**: Use QLoRA or similar for model compression
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA adapters)
- **Modular Design**: Each component clearly separated for explanation
- **Action**: Implement with well-documented, presentation-ready code

### 4. Local Deployment Frameworks

#### 4.1 Ollama (Current)
- **Pros**: Simple, already integrated, good model library
- **Cons**: Limited to available models, less control
- **Action**: Continue using for LLM head, explore for full pipeline

#### 4.2 llama.cpp
- **Pros**: Efficient, supports many models, CPU/GPU
- **Cons**: More setup required, lower-level API
- **Action**: Evaluate for quantized LLM inference
- **Educational Value**: Demonstrates efficient local deployment

#### 4.3 Transformers + PyTorch
- **Pros**: Full control, access to all Hugging Face models
- **Cons**: Higher memory usage, more complex setup
- **Action**: Use for visual encoder and temporal transformer
- **Educational Value**: Full visibility into tokenization/embedding process
- **Resources**: Hugging Face Transformers, PEFT library

#### 4.4 Hybrid Approach (Recommended)
- **Visual Encoder + Temporal Transformer**: PyTorch/Transformers (full control)
- **LLM Head**: Ollama or llama.cpp (efficient inference)
- **Action**: Implement modular pipeline connecting both

### 5. Optimization Strategies (Pragmatic Local Deployment)

#### 5.1 Model Quantization
- **Current**: q2_K (2-bit) in LLaVA 1.6
- **Options**: q4_K, q8_0 (better quality, more resources)
- **QLoRA**: Quantized LoRA for efficient fine-tuning
- **Action**: Benchmark quality vs. speed trade-offs
- **Educational Value**: Explain quantization impact on model size/speed

#### 5.2 Parameter-Efficient Fine-Tuning (PEFT)
- **LoRA Adapters**: Train only small adapter layers
- **Benefits**: Minimal parameters, fast training, local hardware feasible
- **Action**: Implement LoRA adapters for video understanding task
- **Educational Value**: Demonstrate efficient training approach
- **Resources**: Hugging Face PEFT library

#### 5.3 Token Efficiency
- Token merging/pruning strategies
- Reduce visual tokens without quality loss
- Frame sampling optimization
- **Action**: Research and implement token optimization
- **Educational Value**: Show token budget management

#### 5.4 Efficient Video Encoders
- **FastVLM**: Apple's efficient vision encoder (if available)
- **MobileCLIP2**: Lightweight multi-modal encoder
- **CLIP ViT-B/16**: Standard but efficient option
- **Action**: Evaluate encoder efficiency vs. quality
- **Educational Value**: Compare different encoder choices

#### 5.5 Processing Pipeline Optimization
- Frame skipping strategies
- Batch processing
- Async processing
- **Action**: Optimize pipeline for real-time or near-real-time performance

### 6. Computer Vision Enhancements

#### 5.1 Pre-processing
- Frame quality enhancement (super-resolution if needed)
- Noise reduction
- Stabilization
- **Action**: Evaluate impact on LLM performance

#### 5.2 Feature Extraction
- CLIP embeddings for similarity search
- Vision Transformer features
- **Action**: Test if pre-extracted features improve LLM performance

#### 5.3 Motion Detection
- Optical flow for motion detection
- Background subtraction
- **Action**: Use motion to trigger detailed analysis

## Recommended Research Path

### Phase 0: Foundation & Educational Setup (Week 1)
**Goal**: Set up codebase that clearly demonstrates core concepts

1. **Implement Visual Tokenization & Embedding Pipeline**
   - Create frame → patches → embeddings pipeline
   - Visualize tokenization process (show patches)
   - Extract and visualize embeddings (t-SNE plots)
   - **Deliverable**: Code + visualizations for presentation

2. **Implement Temporal Transformer**
   - Build transformer layer for temporal sequences
   - Add attention visualization
   - Show how attention weights change over time
   - **Deliverable**: Working transformer with attention plots

3. **Document Architecture**
   - Create clear architecture diagram
   - Document data flow (frames → tokens → embeddings → attention → output)
   - Prepare explanation of each component
   - **Deliverable**: Architecture documentation for slides

### Phase 1: Quick Wins & Integration (Week 2)
1. **Check Ollama for video models**
   - List available models: `ollama list`
   - Search Ollama library for video-capable models
   - Test LLaVA-NeXT-Video if available

2. **Integrate LLM Head**
   - Connect temporal transformer output to Ollama LLM
   - Test end-to-end pipeline
   - **Deliverable**: Working video → text pipeline

3. **Implement scene change detection**
   - Add basic scene change detector
   - Trigger detailed captions on scene changes
   - Maintain lightweight monitoring otherwise

4. **Create Demo Video**
   - Record simple action video (e.g., pouring water)
   - Test model on demo video
   - Prepare for presentation demonstration

### Phase 2: Model Exploration & Optimization (Week 3-4)
1. **Test TinyLLaVA-Video**
   - Download and test locally
   - Compare with current LLaVA 1.6
   - Evaluate temporal understanding
   - **Educational Note**: Compare architectures for presentation

2. **Evaluate SlowFast-LLaVA Architecture**
   - Test two-stream architecture
   - Measure temporal context capture
   - Compare inference speed
   - **Educational Value**: Demonstrates alternative temporal approach

3. **Implement Quantization & PEFT**
   - Apply QLoRA for model compression
   - Test LoRA adapters for fine-tuning
   - Benchmark performance
   - **Educational Value**: Show practical optimization techniques

4. **Benchmark models**
   - Create test video dataset
   - Compare accuracy, speed, resource usage
   - Document findings
   - **Deliverable**: Performance comparison for presentation

### Phase 3: Advanced Features & Presentation Prep (Week 5-6)
1. **Enhance Temporal Understanding**
   - Frame sequence processing
   - Temporal attention mechanisms
   - Long-term context understanding
   - **Educational Value**: Show advanced temporal reasoning

2. **Add Spatial Understanding**
   - Object detection integration (optional)
   - Spatial relationship queries
   - Scene understanding improvements
   - **Educational Value**: Demonstrate multi-scale understanding

3. **Prepare Presentation Materials**
   - Create slides explaining tokenization, embeddings, transformers
   - Prepare code walkthrough
   - Create visualizations (attention plots, embedding clusters)
   - Record demo video
   - **Deliverable**: Complete presentation materials

4. **Code Documentation**
   - Document key functions with educational comments
   - Create examples showing each concept
   - Prepare code snippets for slides
   - **Deliverable**: Well-documented, presentation-ready codebase

### Phase 4: Refinement & Demo (Week 7-8)
1. **Performance tuning**
   - Optimize for target hardware
   - Reduce latency
   - Improve accuracy
   - Ensure demo runs smoothly

2. **Presentation Refinement**
   - Practice demo
   - Refine explanations
   - Prepare Q&A materials
   - Test on different hardware if needed

3. **Final Integration**
   - Test with real surveillance scenarios (if applicable)
   - Handle edge cases
   - Polish user experience

## Evaluation Criteria

### Performance Metrics
- **Latency**: Time from frame capture to caption
- **Accuracy**: Caption quality and relevance
- **Temporal Understanding**: Ability to track events over time
- **Resource Usage**: CPU, GPU, memory consumption
- **Educational Clarity**: How well the code demonstrates concepts

### Quality Metrics
- **Caption Detail**: Level of detail in descriptions
- **Temporal Accuracy**: Correct understanding of temporal relationships
- **Spatial Understanding**: Ability to describe spatial relationships
- **Action Recognition**: Detection and description of activities
- **Presentation Readiness**: Code clarity, visualizations, documentation

### Educational Metrics
- **Concept Demonstration**: How clearly tokenization/embeddings/transformers are shown
- **Code Readability**: Can others understand the implementation?
- **Visualization Quality**: Are attention plots, embeddings clear?
- **Demo Success**: Does live demo work smoothly?

## Hardware Considerations

### Minimum Requirements (to be determined)
- GPU: [To be tested]
- RAM: [To be tested]
- Storage: [Model sizes vary]

### Optimization Targets
- Real-time processing (30 FPS input, <1s latency)
- Low memory footprint (<8GB if possible)
- Efficient GPU utilization

## Key Questions to Answer

1. **Which video LLM models are available for local deployment?**
   - Check Ollama library
   - Research Hugging Face models
   - Test availability and compatibility

2. **What is the best approach for temporal understanding?**
   - Frame-by-frame vs. sequence processing
   - Optimal frame sampling rate
   - Temporal aggregation methods

3. **How to balance accuracy vs. speed?**
   - Quantization trade-offs
   - Model size vs. performance
   - Processing frequency optimization

4. **What spatio-temporal features are most valuable?**
   - Object tracking
   - Action recognition
   - Scene understanding
   - Temporal localization

## Next Steps

1. **Immediate Actions**:
   - Check Ollama for available video models
   - Set up test environment
   - Create benchmark dataset

2. **Research Tasks**:
   - Deep dive into each model's architecture
   - Test local deployment feasibility
   - Compare performance characteristics

3. **Implementation Tasks**:
   - Start with Phase 1 quick wins
   - Iterate based on findings
   - Document all experiments

## Resources

### Model Repositories
- Ollama: https://ollama.com/library
- Hugging Face: https://huggingface.co/models
- Awesome Video LLMs: https://github.com/zyayoung/Awesome-Video-LLMs

### Papers & Documentation
- LongVLM: https://arxiv.org/abs/2404.03384
- SlowFast-LLaVA: https://arxiv.org/abs/2407.15841
- TimeMarker: https://arxiv.org/abs/2411.18211
- TinyLLaVA-Video: https://arxiv.org/abs/2501.15513

### Frameworks
- Ollama: https://ollama.com
- llama.cpp: https://github.com/ggerganov/llama.cpp
- Transformers: https://huggingface.co/docs/transformers
- PEFT (Parameter-Efficient Fine-Tuning): https://huggingface.co/docs/peft
- Hugging Face Tokenizers: https://huggingface.co/docs/tokenizers

### Educational Resources
- Vision Transformers Explained: https://arxiv.org/abs/2010.11929
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- CLIP Paper: https://arxiv.org/abs/2103.00020
- Video Understanding Surveys: https://github.com/zyayoung/Awesome-Video-LLMs

## Presentation Structure (Aligned with AI-mode.md)

### Part 1: The Building Blocks of Understanding
- **Slide 1: Tokenization** - Frame → Patches → Tokens
- **Slide 2: Embeddings** - Tokens → Vectors (with t-SNE visualization)
- **Slide 3: Transformers** - Attention mechanism (with attention plots)

### Part 2: A Pragmatic Video Understanding Model
- **Slide 4: Project Overview** - Local LLM for spatio-temporal understanding
- **Slide 5: Model Architecture** - Visual Encoder → Temporal Transformer → LLM Head
- **Slide 6: Implementation Details** - Quantization (QLoRA), PEFT, Inference Pipeline
- **Slide 7: Live Demo** - Video input → Model query → Text output

## Conclusion

This research plan provides a structured approach that serves dual purposes:
1. **Educational**: Create a presentation explaining tokenization, embeddings, and transformers through video understanding
2. **Practical**: Build a working local video understanding model demonstrating these concepts

The focus on local deployment ensures privacy and control while providing a pragmatic example of modern deep learning techniques. The phased approach allows for iterative development, with each phase building both technical capability and educational materials for the presentation.

