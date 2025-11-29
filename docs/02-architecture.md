# Architecture Overview

## Project Overview

This project combines **educational presentation goals** with **practical implementation** of a video understanding system using locally served LLMs. The focus is on demonstrating fundamental deep learning concepts (tokenization, embeddings, transformers) through a working video understanding model.

## Dual Purpose

### 1. Educational Presentation
- Explain **tokenization**: How video frames become tokens
- Explain **embeddings**: How tokens become meaningful vectors
- Explain **transformers**: How attention mechanisms process sequences
- Provide clear, demonstrable code examples

### 2. Practical Implementation
- Build working video understanding model
- Run entirely on local hardware
- Demonstrate spatio-temporal understanding
- Show quantization and efficiency techniques

## Architecture Design

### Proposed Architecture (Presentation-Ready)

```
Video Frames
    ↓
[Visual Encoder: CLIP/ViT]
    ├─ Tokenization: Frames → Patches → Tokens
    └─ Embedding: Tokens → Dense Vectors
    ↓
[Temporal Transformer]
    ├─ Self-Attention: Temporal relationships
    └─ Positional Encoding: Temporal order
    ↓
[LLM Head: Quantized LLM]
    └─ Text Generation: Visual → Language
    ↓
Text Output
```

### Educational Components

1. **Tokenization Module**
   - Shows: Frame → Patches → Tokens
   - Visualization: Frame with patch grid
   - Code: Clear, commented implementation

2. **Embedding Module**
   - Shows: Tokens → Vectors
   - Visualization: t-SNE plot of embeddings
   - Code: Embedding extraction and visualization

3. **Transformer Module**
   - Shows: Self-attention mechanism
   - Visualization: Attention heatmaps
   - Code: Temporal transformer with attention extraction

4. **LLM Integration**
   - Shows: Visual understanding → Language
   - Demo: Video input → Text output
   - Code: End-to-end pipeline

## Component Breakdown

### 1. Visual Encoder (Tokenization + Embedding)
**Purpose**: Convert video frames into token embeddings

**Implementation Options**:
- **CLIP ViT-B/16**: Standard, well-documented
- **MobileCLIP2**: Lightweight, efficient
- **Custom ViT**: Full control, educational value

**Key Code Components**:
```python
# Frame tokenization
def tokenize_frame(frame, patch_size=16):
    """
    Convert frame to patches (tokens)
    Educational: Shows how images become tokens
    """
    patches = extract_patches(frame, patch_size)
    return patches

# Frame embedding
def embed_frame(patches, encoder):
    """
    Convert patches to embeddings
    Educational: Shows tokens → vectors transformation
    """
    embeddings = encoder(patches)
    return embeddings
```

**Visualizations Needed**:
- Frame with patch grid overlay
- Embedding vectors (dimensionality reduction for visualization)
- t-SNE plot of frame embeddings

### 2. Temporal Transformer
**Purpose**: Process sequence of frame embeddings with attention

**Implementation**:
```python
class TemporalTransformer(nn.Module):
    """
    Transformer for temporal sequence processing
    Educational: Demonstrates self-attention mechanism
    """
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.transformer = nn.TransformerEncoder(...)
        self.positional_encoding = PositionalEncoding(...)
    
    def forward(self, frame_embeddings):
        # Add positional encoding
        # Apply self-attention
        # Return temporal context
        pass
```

**Key Features**:
- Self-attention across temporal dimension
- Positional encoding for temporal order
- Attention visualization (heatmaps)

**Visualizations Needed**:
- Attention weight heatmaps (frame × frame)
- Temporal attention patterns
- How attention changes with different queries

### 3. LLM Head
**Purpose**: Generate text from visual understanding

**Implementation Options**:
- **Ollama Integration**: Simple, already set up
- **llama.cpp**: More control, efficient
- **Hugging Face Transformers**: Full visibility

**Key Code Components**:
```python
def generate_caption(temporal_features, query, llm_client):
    """
    Generate text response from visual features
    Educational: Shows visual → language bridge
    """
    # Format prompt with visual context
    prompt = format_prompt(temporal_features, query)
    # Query LLM
    response = llm_client.generate(prompt)
    return response
```

## Implementation Phases

### Phase 0: Foundation (Week 1)
- Implement tokenization and embedding pipeline
- Build temporal transformer
- Create visualizations
- **Deliverable**: Working components with clear educational value

### Phase 1: Integration (Week 2)
- Connect to LLM (Ollama)
- Test end-to-end pipeline
- Optimize for performance
- **Deliverable**: Working demo

### Phase 2: Enhancement (Week 3-4)
- Test alternative models
- Implement quantization/PEFT
- Benchmark performance
- **Deliverable**: Optimized, presentation-ready system

### Phase 3: Presentation Prep (Week 5-6)
- Create slides
- Prepare code walkthrough
- Record demo
- **Deliverable**: Complete presentation materials

## Technical Stack

### Core Libraries
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Pre-trained models, tokenizers
- **Ollama**: Local LLM inference
- **OpenCV**: Video processing
- **Matplotlib/Seaborn**: Visualizations

### Models
- **Visual Encoder**: CLIP ViT-B/16 or MobileCLIP2
- **Temporal Transformer**: Custom implementation
- **LLM**: Ollama models (llava:7b-v1.6-mistral-q2_K or newer)

### Optimization
- **Quantization**: QLoRA, model quantization
- **PEFT**: LoRA adapters for efficient fine-tuning
- **Efficient Encoders**: MobileCLIP2, FastVLM

## Code Structure

```
vid-sense/
├── src/
│   ├── tokenization.py      # Frame → patches → tokens
│   ├── embeddings.py         # Tokens → embeddings
│   ├── temporal_transformer.py  # Transformer for sequences
│   ├── llm_integration.py   # LLM connection
│   ├── pipeline.py          # End-to-end pipeline
│   └── visualizations.py    # All visualization code
├── notebooks/
│   ├── 01_tokenization_demo.ipynb
│   ├── 02_embeddings_demo.ipynb
│   └── 03_attention_demo.ipynb
├── demos/
│   ├── demo_video.mp4       # Test video
│   └── demo_queries.txt     # Example queries
├── examples/
│   └── reference-implementation.py  # Reference code
├── presentations/
│   └── slides.md            # Presentation materials
└── requirements.txt
```

## Success Criteria

### Educational Success
- [ ] Clear explanation of tokenization, embeddings, transformers
- [ ] Helpful visualizations
- [ ] Understandable code
- [ ] Smooth demo

### Technical Success
- [ ] Working video understanding model
- [ ] Runs on local hardware
- [ ] Acceptable performance (latency, accuracy)
- [ ] Spatio-temporal understanding demonstrated

### Presentation Success
- [ ] Engaging presentation
- [ ] Clear technical explanations
- [ ] Successful live demo
- [ ] Well-received by audience

