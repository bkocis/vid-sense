# Vid-Sense: Video Understanding with Local LLMs

A project combining **educational presentation goals** with **practical implementation** of a video understanding system using locally served LLMs. This project demonstrates fundamental deep learning concepts (tokenization, embeddings, transformers) through a working video understanding model.

## 🎯 Project Goals

1. **Educational**: Explain tokenization, embeddings, and transformers through video understanding
2. **Practical**: Build a working local video understanding model with spatio-temporal reasoning
3. **Presentation-Ready**: Code and demo that clearly demonstrates concepts

## 📚 Documentation Structure

### Getting Started
- **[Quick Start Guide](docs/01-getting-started.md)** - Actionable next steps to begin implementation
- **[Architecture Overview](docs/02-architecture.md)** - System design and component breakdown

### Implementation
- **[Implementation Guide](docs/03-implementation.md)** - Step-by-step implementation with code examples
- **[Research & Planning](docs/04-research.md)** - Research findings, model comparisons, and recommendations

### Examples

### Presentation
- **[Slide Deck](presentations/slides.md)** - Complete presentation materials

## 🏗️ Architecture Overview

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

## 🚀 Quick Start

1. **Review the documentation**:
   - Start with [Quick Start Guide](docs/01-getting-started.md)
   - Understand the [Architecture](docs/02-architecture.md)
   - Follow the [Implementation Guide](docs/03-implementation.md)

2. **Check your setup**:
   ```bash
   ollama list
   ollama search video
   ```


## 📋 Key Features

### Educational Value
- ✅ Clear demonstration of tokenization, embeddings, transformers
- ✅ Visualizations (attention plots, embedding clusters)
- ✅ Well-documented, presentation-ready code
- ✅ Step-by-step explanations

### Practical Value
- ✅ Runs entirely on local hardware
- ✅ Efficient (quantization, PEFT)
- ✅ Spatio-temporal understanding
- ✅ Real-time or near-real-time performance

## 🛠️ Technical Stack

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

## 📖 Project Status

**Current Phase**: Planning & Research Complete  
**Next Phase**: Foundation Implementation  
**Timeline**: 6-8 weeks to presentation-ready

## 🔗 Resources

### Model Repositories
- [Ollama Library](https://ollama.com/library)
- [Hugging Face Models](https://huggingface.co/models)
- [Awesome Video LLMs](https://github.com/zyayoung/Awesome-Video-LLMs)

### Key Papers
- Vision Transformers: https://arxiv.org/abs/2010.11929
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- CLIP: https://arxiv.org/abs/2103.00020
- SlowFast-LLaVA: https://arxiv.org/abs/2407.15841
- LongVLM: https://arxiv.org/abs/2404.03384

## 📝 License

This project is for educational and research purposes.

---

**Note**: This project builds upon the [ollama-home-surveillance](https://github.com/bkocis/home-surveillance-with-multimodal-llms) project, extending it with temporal understanding and educational components.
