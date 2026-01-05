# Transformer Visualizer 🔮

An interactive web-based visualization tool for understanding the Transformer architecture from "Attention is All You Need" (Vaswani et al., 2017).

![Transformer Visualization](https://img.shields.io/badge/Paper-Attention%20is%20All%20You%20Need-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-orange)
![Interactive](https://img.shields.io/badge/UI-Interactive-green)

## Features

### 📚 Comprehensive Code Documentation
- **Detailed comments** in all Python files with paper section references
- **Mathematical formulas** explained inline (e.g., "Section 3.2, Equation 1")
- **Shape transformations** documented at every step

### 🎨 Interactive Visualization UI
- **Live input processing** - Enter any sentence and see it flow through the Transformer
- **6 visualization phases**:
  1. Tokenization
  2. Input Embedding (with √d_model scaling)
  3. Positional Encoding (sin/cos waves)
  4. Encoder (Multi-Head Attention + FFN)
  5. Decoder (Masked Attention + Cross-Attention + FFN)
  6. Output Projection (Linear + Softmax)

### 🐍 Code-to-UI Mapping
- **Expandable code panels** showing Python implementation
- **Direct links** from UI layers to `model.py` classes
- **Class navigation panel** for quick jumping

### 📊 Visual Features
- **Attention heatmaps** for self-attention and cross-attention
- **Shape tracking** at every transformation (e.g., `(5,) → (5, 512)`)
- **Formula boxes** with paper references
- **Color-coded** Q/K/V matrices

## Quick Start

### View the Visualization

1. Clone this repository:
```bash
git clone https://github.com/Shabzsagar16/transformer-visual.git
cd transformer-visual
```

2. Open the visualization:
```bash
open transformer_viz/index.html
```
Or simply drag `transformer_viz/index.html` into your browser.

3. Enter a sentence and click "Process Through Transformer"

### Train the Model

```bash
# Install dependencies
pip install torch torchvision torchaudio
pip install tokenizers datasets

# Run training
python train.py
```

## Project Structure

```
pytorch-transformer/
├── model.py              # Transformer implementation (fully commented)
├── dataset.py            # Data loading and preprocessing
├── config.py             # Configuration and hyperparameters
├── train.py              # Training script
└── transformer_viz/      # Interactive visualization
    ├── index.html        # Main UI
    ├── styles.css        # Styling
    └── script.js         # Visualization logic
```

## Code Documentation

All classes are documented with:
- Paper section references
- Mathematical formulas
- Shape transformations
- Implementation notes

### Key Classes

| Class | Location | Paper Reference |
|-------|----------|-----------------|
| `InputEmbeddings` | model.py:73 | Section 3.4 |
| `PositionalEncoding` | model.py:101 | Section 3.5 |
| `MultiHeadAttentionBlock` | model.py:202 | Section 3.2, Eq. 1 |
| `FeedForwardBlock` | model.py:44 | Section 3.3 |
| `EncoderBlock` | model.py:341 | Section 3.1 |
| `DecoderBlock` | model.py:404 | Section 3.1 |
| `ProjectionLayer` | model.py:480 | Output Layer |

## Visualization Screenshots

The UI shows:
- Token-by-token processing
- Embedding transformations
- Attention score heatmaps
- Shape changes at each layer
- Python code for each component

## Model Configuration

Default hyperparameters (from paper):
- `d_model`: 512
- `N` (layers): 6
- `h` (heads): 8
- `d_k` = `d_v`: 64
- `d_ff`: 2048
- Dropout: 0.1

## Paper Reference

This implementation is based on:

**"Attention is All You Need"**  
Vaswani et al., 2017  
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## Credits

- Original PyTorch implementation: [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer)
- Visualization and documentation enhancements: This repository

## License

MIT License - See original repository for details

## Contributing

Feel free to open issues or submit pull requests for:
- Additional visualization features
- More detailed explanations
- Bug fixes
- Documentation improvements

---

Made with ❤️ for understanding Transformers
