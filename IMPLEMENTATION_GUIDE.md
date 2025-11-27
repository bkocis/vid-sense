# Implementation Guide: Video Understanding Model for Presentation

## Overview

This guide provides a step-by-step implementation plan for building a video understanding model that demonstrates tokenization, embeddings, and transformers in practice. The implementation will be presentation-ready with clear code, visualizations, and explanations.

## Architecture Design

### High-Level Architecture

```
Video Input (Frames)
    ↓
[Visual Encoder: CLIP/ViT]
    ├─ Tokenization: Frames → Patches → Tokens
    └─ Embedding: Tokens → Dense Vectors
    ↓
[Temporal Transformer]
    ├─ Self-Attention: Temporal relationships
    └─ Positional Encoding: Temporal order
    ↓
[LLM Head: Quantized LLM via Ollama]
    └─ Text Generation: Visual understanding → Language
    ↓
Text Output (Caption/Answer)
```

### Component Breakdown

#### 1. Visual Encoder (Tokenization + Embedding)
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

#### 2. Temporal Transformer
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

#### 3. LLM Head
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
**Goal**: Build components that clearly demonstrate core concepts

#### Step 1: Visual Tokenization & Embedding
**Files to Create**:
- `src/tokenization.py`: Frame → patches → tokens
- `src/embeddings.py`: Tokens → embeddings
- `src/visualizations.py`: Plot embeddings, patches

**Tasks**:
1. Implement frame patch extraction
2. Integrate CLIP or ViT encoder
3. Extract embeddings from frames
4. Create t-SNE visualization of embeddings
5. Document with educational comments

**Deliverable**: 
- Working tokenization/embedding pipeline
- Visualization showing patches and embeddings
- Code ready for presentation slides

#### Step 2: Temporal Transformer
**Files to Create**:
- `src/temporal_transformer.py`: Transformer implementation
- `src/attention_viz.py`: Attention visualization

**Tasks**:
1. Implement transformer encoder for temporal sequences
2. Add positional encoding
3. Extract attention weights
4. Create attention heatmap visualization
5. Test on frame sequences

**Deliverable**:
- Working temporal transformer
- Attention visualization code
- Example attention plots

#### Step 3: Integration with LLM
**Files to Create**:
- `src/llm_integration.py`: Connect to Ollama/llama.cpp
- `src/pipeline.py`: End-to-end pipeline

**Tasks**:
1. Connect temporal features to LLM
2. Format prompts with visual context
3. Test end-to-end pipeline
4. Handle quantization if needed

**Deliverable**:
- Working video → text pipeline
- Demo-ready code

### Phase 1: Optimization (Week 2)
**Goal**: Make it efficient and presentation-ready

#### Step 1: Quantization
- Apply QLoRA if fine-tuning needed
- Test different quantization levels
- Document trade-offs

#### Step 2: Performance Optimization
- Optimize frame sampling
- Batch processing
- Async if needed

#### Step 3: Demo Preparation
- Record demo video
- Prepare test queries
- Ensure smooth execution

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
├── presentations/
│   └── slides/              # Presentation materials
└── requirements.txt
```

## Key Implementation Details

### Tokenization Implementation

```python
import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

class VideoTokenizer:
    """
    Tokenizes video frames into patches and embeddings
    Educational: Clear demonstration of tokenization process
    """
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
    
    def tokenize_frame(self, frame):
        """
        Convert frame to tokens (patches)
        Returns: patches, patch_embeddings
        """
        # Process frame
        inputs = self.processor(images=frame, return_tensors="pt")
        
        # Get patch embeddings
        outputs = self.model.get_image_features(**inputs)
        
        return outputs  # These are the "tokens" (patch embeddings)
    
    def visualize_patches(self, frame, patch_size=16):
        """Visualize how frame is divided into patches"""
        # Implementation for visualization
        pass
```

### Temporal Transformer Implementation

```python
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Add positional encoding for temporal order"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        # Standard sinusoidal positional encoding
        pass

class TemporalTransformer(nn.Module):
    """
    Transformer for processing temporal sequences
    Educational: Shows self-attention in action
    """
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, frame_embeddings, return_attention=False):
        """
        Process sequence of frame embeddings
        frame_embeddings: [batch, seq_len, d_model]
        """
        # Add positional encoding
        x = self.pos_encoder(frame_embeddings)
        
        # Apply transformer
        output = self.transformer(x)
        
        if return_attention:
            # Extract attention weights for visualization
            attention_weights = self._get_attention_weights(x)
            return output, attention_weights
        
        return output
```

### LLM Integration

```python
import ollama

class LLMHead:
    """
    Connects visual understanding to language generation
    Educational: Shows visual → language bridge
    """
    def __init__(self, model_name="llava:7b-v1.6-mistral-q2_K"):
        self.model_name = model_name
    
    def generate_response(self, temporal_features, query, frame_images):
        """
        Generate text response from visual features
        """
        # Format prompt
        prompt = self._format_prompt(temporal_features, query)
        
        # Query Ollama with frames
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': frame_images  # Send actual frames
                }
            ]
        )
        
        return response['message']['content']
    
    def _format_prompt(self, features, query):
        """Format prompt with visual context"""
        # Could include feature summary or just rely on images
        return f"Based on these video frames: {query}"
```

## Visualization Requirements

### 1. Tokenization Visualization
- Show frame divided into patches
- Highlight patch boundaries
- Show patch → token mapping

### 2. Embedding Visualization
- t-SNE plot of frame embeddings
- Show similar frames clustering
- Temporal progression in embedding space

### 3. Attention Visualization
- Heatmap: attention weights across frames
- Show which frames attend to which
- Query-specific attention patterns

### 4. Pipeline Visualization
- End-to-end flow diagram
- Data transformations at each step
- Real-time during demo

## Testing & Demo

### Test Video
- Simple, repeatable action (e.g., pouring water)
- Clear, well-lit
- 5-10 seconds duration
- Multiple frames for temporal understanding

### Test Queries
- "What action is taking place?"
- "What objects are visible?"
- "Describe the sequence of events"
- "What happens first, then what?"

### Demo Script
1. Load test video
2. Show tokenization (patches)
3. Show embeddings (t-SNE)
4. Process through transformer
5. Show attention weights
6. Generate caption/answer
7. Display output

## Resources

### Libraries Needed
```txt
torch
transformers
ollama
opencv-python
matplotlib
seaborn
scikit-learn  # for t-SNE
numpy
pillow
```

### Model Resources
- CLIP: `openai/clip-vit-base-patch16`
- Alternative: `mobileclip` for lighter option
- LLM: Existing Ollama models

## Next Steps

1. **Set up project structure**
2. **Implement tokenization module**
3. **Implement embedding extraction**
4. **Create visualizations**
5. **Build temporal transformer**
6. **Integrate with LLM**
7. **Test end-to-end**
8. **Prepare demo**

This implementation guide provides a clear path from concept to working demo, with each component designed to be educational and presentation-ready.

