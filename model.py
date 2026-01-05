# PyTorch Transformer Implementation
# Based on "Attention is All You Need" (Vaswani et al., 2017)
# Paper: https://arxiv.org/abs/1706.03762

import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    """
    Layer Normalization as described in Section 3.1 of the paper.
    
    Normalizes the features across the embedding dimension (d_model) for each token independently.
    Unlike Batch Normalization which normalizes across the batch, Layer Normalization normalizes
    across features, making it more suitable for sequence data and variable-length sequences.
    
    Formula: LayerNorm(x) = α * (x - μ) / (σ + ε) + β
    where:
        μ = mean across features (d_model dimension)
        σ = standard deviation across features
        α (alpha) = learnable scale parameter (initialized to 1)
        β (bias) = learnable shift parameter (initialized to 0)
        ε (eps) = small constant for numerical stability
    """

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scale parameter (α)
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable shift parameter (β)

    def forward(self, x):
        # Input: x with shape (batch, seq_len, features)
        # Normalize across the feature/embedding dimension (last dimension)
        
        mean = x.mean(dim=-1, keepdim=True)  # Shape: (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)    # Shape: (batch, seq_len, 1)
        
        # Apply normalization: (x - μ) / (σ + ε)
        # Then apply learnable affine transformation: α * normalized + β
        # eps prevents division by zero when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    Position-wise Feed-Forward Network as described in Section 3.3 of the paper.
    
    Consists of two linear transformations with a ReLU activation in between.
    Applied to each position separately and identically (same weights for all positions).
    
    Formula: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    Architecture:
        - First layer: d_model → d_ff (expansion, typically d_ff = 2048 for d_model = 512)
        - ReLU activation
        - Dropout for regularization
        - Second layer: d_ff → d_model (projection back to model dimension)
    
    In the paper: d_model = 512, d_ff = 2048
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W₁ and b₁
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W₂ and b₂

    def forward(self, x):
        # Shape transformations:
        # (batch, seq_len, d_model) → (batch, seq_len, d_ff) → (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    """
    Learned Embeddings as described in Section 3.4 of the paper.
    
    Converts input tokens (integers) to dense vectors of dimension d_model.
    Uses learned embeddings similar to other sequence transduction models.
    
    The embeddings are scaled by sqrt(d_model) as mentioned in Section 3.4:
    "In the embedding layers, we multiply those weights by sqrt(d_model)"
    
    This scaling helps maintain consistent variance across different model dimensions
    and balances the magnitude of embeddings with positional encodings.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Input: (batch, seq_len) - token indices
        # Output: (batch, seq_len, d_model) - embedded vectors
        
        # Multiply by sqrt(d_model) as specified in Section 3.4 of the paper
        # This scaling is important for training stability
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    """
    Positional Encoding as described in Section 3.5 of the paper.
    
    Since the Transformer has no recurrence or convolution, it needs explicit positional information.
    Uses sinusoidal functions of different frequencies to encode position information.
    
    Formulas from the paper:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where:
        pos = position in the sequence (0 to seq_len-1)
        i = dimension index (0 to d_model/2 - 1)
        2i = even indices use sine
        2i+1 = odd indices use cosine
    
    Benefits of this approach:
        - Allows the model to easily learn to attend by relative positions
        - Can extrapolate to sequence lengths longer than those seen during training
        - Each dimension corresponds to a sinusoid with wavelengths forming geometric progression
          from 2π to 10000 · 2π
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model) to hold positional encodings
        pe = torch.zeros(seq_len, d_model)
        
        # Create a vector of positions: [0, 1, 2, ..., seq_len-1]
        # Shape: (seq_len, 1) - unsqueeze for broadcasting
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create the division term for the positional encoding formula
        # This creates: [1, 1/10000^(2/d_model), 1/10000^(4/d_model), ..., 1/10000^((d_model-2)/d_model)]
        # We use exp and log for numerical stability: 10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        # Shape: (d_model/2,) - only need half since we'll use for both sin and cos
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (0, 2, 4, ...)
        # position * div_term broadcasts to (seq_len, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (seq_len, d_model) -> (1, seq_len, d_model)
        # This allows broadcasting across batches
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but should be saved with the model)
        # Buffers are tensors that should be part of the model state but are not trainable
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input embeddings
        # x shape: (batch, seq_len, d_model)
        # self.pe shape: (1, max_seq_len, d_model)
        
        # Add positional encoding up to the sequence length of x
        # Set requires_grad to False since positional encodings are fixed (not learned)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        
        # Apply dropout for regularization
        return self.dropout(x)

class ResidualConnection(nn.Module):
    """
    Residual Connection with Layer Normalization as described in Section 3.1 of the paper.
    
    Each sub-layer in the encoder and decoder has a residual connection around it,
    followed by layer normalization. The paper describes this as:
    "We apply dropout to the output of each sub-layer, before it is added to the 
    sub-layer input and normalized."
    
    Formula: LayerNorm(x + Sublayer(x))
    
    However, this implementation uses Pre-Layer Normalization (Pre-LN):
    Formula: x + Dropout(Sublayer(LayerNorm(x)))
    
    Pre-LN has been shown to be more stable for training and is commonly used
    in modern Transformer implementations.
    """
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        # Apply Pre-Layer Normalization:
        # 1. Normalize the input
        # 2. Pass through the sublayer (attention or feed-forward)
        # 3. Apply dropout
        # 4. Add to original input (residual connection)
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention as described in Section 3.2 of the paper.
    
    Multi-head attention allows the model to jointly attend to information from different
    representation subspaces at different positions. This is the core mechanism that allows
    the Transformer to capture complex relationships in the input.
    
    Scaled Dot-Product Attention Formula (Equation 1 in the paper):
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    where:
        Q = Query matrix
        K = Key matrix
        V = Value matrix
        d_k = dimension of keys (used for scaling)
    
    Multi-Head Attention Formula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    
    Linear projections:
        W^Q_i ∈ ℝ^(d_model × d_k) - Query projection for head i
        W^K_i ∈ ℝ^(d_model × d_k) - Key projection for head i
        W^V_i ∈ ℝ^(d_model × d_v) - Value projection for head i
        W^O ∈ ℝ^(h·d_v × d_model) - Output projection
    
    In the paper: h = 8 parallel attention heads, d_k = d_v = d_model/h = 64
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding dimension (e.g., 512)
        self.h = h              # Number of attention heads (e.g., 8)
        
        # Ensure d_model is divisible by h so we can split into h heads
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h  # Dimension of each head: d_k = d_v = d_model / h = 64
        
        # Linear projection matrices
        # Instead of separate W^Q_i, W^K_i, W^V_i for each head, we use single large matrices
        # and split the result into h heads. This is more efficient.
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # W^Q: (d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # W^K: (d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # W^V: (d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # W^O: (d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute Scaled Dot-Product Attention (Figure 2 in the paper).
        
        Formula: Attention(Q, K, V) = softmax(QK^T / √d_k) V
        
        The scaling by √d_k prevents the dot products from growing too large,
        which would push the softmax into regions with extremely small gradients.
        
        Args:
            query: Query tensor (batch, h, seq_len, d_k)
            key: Key tensor (batch, h, seq_len, d_k)
            value: Value tensor (batch, h, seq_len, d_k)
            mask: Mask tensor to prevent attention to certain positions
            dropout: Dropout layer for regularization
        
        Returns:
            attention_output: Weighted values (batch, h, seq_len, d_k)
            attention_scores: Attention weights for visualization (batch, h, seq_len, seq_len)
        """
        d_k = query.shape[-1]
        
        # Step 1: Compute attention scores - QK^T / √d_k
        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Step 2: Apply mask (if provided)
        # Masking is used for two purposes:
        # - Padding mask: prevent attention to padding tokens (mask == 0)
        # - Causal mask: prevent attention to future positions in decoder (mask == 0)
        if mask is not None:
            # Set masked positions to large negative value
            # After softmax, these positions will have ~0 probability
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Step 3: Apply softmax to get attention weights
        # Softmax normalizes scores so they sum to 1 along the last dimension
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)
        
        # Step 4: Apply dropout for regularization
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # Step 5: Multiply by values to get weighted representation
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
        attention_output = attention_scores @ value
        
        # Return both the output and attention scores (scores useful for visualization)
        return attention_output, attention_scores

    def forward(self, q, k, v, mask):
        """
        Forward pass for Multi-Head Attention.
        
        Args:
            q: Query input (batch, seq_len, d_model)
            k: Key input (batch, seq_len, d_model)
            v: Value input (batch, seq_len, d_model)
            mask: Attention mask
        
        Returns:
            Output after multi-head attention (batch, seq_len, d_model)
        """
        # Step 1: Apply linear projections Q, K, V
        query = self.w_q(q)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k)    # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # Step 2: Split into multiple heads
        # Reshape from (batch, seq_len, d_model) to (batch, seq_len, h, d_k)
        # Then transpose to (batch, h, seq_len, d_k) for parallel attention computation
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Step 3: Compute attention for all heads in parallel
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Step 4: Concatenate all heads
        # Transpose back: (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k)
        # Reshape to concatenate: (batch, seq_len, h, d_k) -> (batch, seq_len, h * d_k)
        # Note: h * d_k = d_model
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Step 5: Apply final linear projection W^O
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)

class EncoderBlock(nn.Module):
    """
    Single Encoder Block as described in Section 3.1 and Figure 1 (left) of the paper.
    
    Each encoder block consists of two sub-layers:
    1. Multi-Head Self-Attention mechanism
    2. Position-wise Feed-Forward Network
    
    Each sub-layer has:
    - A residual connection around it
    - Layer normalization
    
    Formula for each sub-layer:
        output = x + Dropout(Sublayer(LayerNorm(x)))
    
    The encoder is composed of a stack of N=6 identical encoder blocks.
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Two residual connections: one for attention, one for feed-forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Sub-layer 1: Multi-Head Self-Attention
        # Query, Key, and Value all come from the same source (x)
        # This allows each position to attend to all positions in the input
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        # Sub-layer 2: Position-wise Feed-Forward Network
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x
    
class Encoder(nn.Module):
    """
    Complete Encoder as described in Section 3.1 of the paper.
    
    The encoder is composed of a stack of N=6 identical encoder blocks.
    Each block processes the input sequentially, with the output of one block
    becoming the input to the next.
    
    After all encoder blocks, a final layer normalization is applied.
    This final normalization helps stabilize the encoder output before it's
    passed to the decoder.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # Stack of N encoder blocks
        self.norm = LayerNormalization(features)  # Final layer normalization

    def forward(self, x, mask):
        # Pass input through each encoder block sequentially
        # Each block applies self-attention and feed-forward transformations
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final layer normalization
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    Single Decoder Block as described in Section 3.1 and Figure 1 (right) of the paper.
    
    Each decoder block consists of three sub-layers:
    1. Masked Multi-Head Self-Attention mechanism
       - Allows each position to attend only to earlier positions (causal/autoregressive)
       - Prevents information flow from future tokens during training
    
    2. Multi-Head Cross-Attention mechanism (Encoder-Decoder Attention)
       - Query comes from the previous decoder layer
       - Key and Value come from the encoder output
       - Allows decoder to attend to all positions in the input sequence
    
    3. Position-wise Feed-Forward Network
    
    Each sub-layer has a residual connection and layer normalization.
    The decoder is composed of a stack of N=6 identical decoder blocks.
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block      # Masked self-attention
        self.cross_attention_block = cross_attention_block    # Encoder-decoder attention
        self.feed_forward_block = feed_forward_block
        # Three residual connections: self-attention, cross-attention, feed-forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Sub-layer 1: Masked Multi-Head Self-Attention
        # Query, Key, and Value all come from the decoder input (x)
        # tgt_mask prevents attending to future positions (autoregressive property)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        
        # Sub-layer 2: Multi-Head Cross-Attention (Encoder-Decoder Attention)
        # Query: from decoder (x)
        # Key and Value: from encoder output
        # This allows the decoder to attend to the entire input sequence
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        
        # Sub-layer 3: Position-wise Feed-Forward Network
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        return x
    
class Decoder(nn.Module):
    """
    Complete Decoder as described in Section 3.1 of the paper.
    
    The decoder is composed of a stack of N=6 identical decoder blocks.
    Each block processes the target sequence while also attending to the encoder output.
    
    The decoder is autoregressive: during training and inference, each position can only
    attend to earlier positions in the output sequence. This is enforced by the causal mask.
    
    After all decoder blocks, a final layer normalization is applied before the
    projection to vocabulary.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # Stack of N decoder blocks
        self.norm = LayerNormalization(features)  # Final layer normalization

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Pass through each decoder block sequentially
        # Each block applies:
        # 1. Masked self-attention on target sequence
        # 2. Cross-attention with encoder output
        # 3. Feed-forward network
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Apply final layer normalization
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    Final Linear Projection Layer.
    
    Projects the decoder output from d_model dimensions to vocab_size dimensions.
    This produces logits for each token in the vocabulary.
    
    During training, these logits are passed through softmax and compared with
    the target tokens using cross-entropy loss.
    
    During inference, the token with the highest logit is selected (greedy decoding)
    or sampling/beam search is used for better results.
    """

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # Input: (batch, seq_len, d_model)
        # Output: (batch, seq_len, vocab_size)
        # Each position gets a distribution over the entire vocabulary
        return self.proj(x)
    
class Transformer(nn.Module):
    """
    Complete Transformer Model as described in "Attention is All You Need" (Vaswani et al., 2017).
    
    The Transformer follows an encoder-decoder architecture:
    
    Encoder:
    - Input Embedding + Positional Encoding
    - Stack of N=6 encoder blocks (self-attention + feed-forward)
    - Processes the source sequence
    
    Decoder:
    - Output Embedding + Positional Encoding
    - Stack of N=6 decoder blocks (masked self-attention + cross-attention + feed-forward)
    - Processes the target sequence while attending to encoder output
    
    Final Layer:
    - Linear projection to vocabulary size
    - Produces probability distribution over target vocabulary
    
    Key innovations from the paper:
    - Relies entirely on attention mechanisms (no recurrence or convolution)
    - Parallel processing of sequences (unlike RNNs)
    - Multi-head attention for different representation subspaces
    - Positional encoding to inject sequence order information
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed              # Source embedding layer
        self.tgt_embed = tgt_embed              # Target embedding layer
        self.src_pos = src_pos                  # Source positional encoding
        self.tgt_pos = tgt_pos                  # Target positional encoding
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Encode the source sequence.
        
        Args:
            src: Source token indices (batch, seq_len)
            src_mask: Mask for source padding (batch, 1, 1, seq_len)
        
        Returns:
            Encoder output (batch, seq_len, d_model)
        """
        # Step 1: Embed source tokens
        src = self.src_embed(src)
        # Step 2: Add positional encoding
        src = self.src_pos(src)
        # Step 3: Pass through encoder stack
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Decode the target sequence given encoder output.
        
        Args:
            encoder_output: Output from encoder (batch, src_seq_len, d_model)
            src_mask: Mask for source padding
            tgt: Target token indices (batch, tgt_seq_len)
            tgt_mask: Causal mask for target (prevents looking at future tokens)
        
        Returns:
            Decoder output (batch, tgt_seq_len, d_model)
        """
        # Step 1: Embed target tokens
        tgt = self.tgt_embed(tgt)
        # Step 2: Add positional encoding
        tgt = self.tgt_pos(tgt)
        # Step 3: Pass through decoder stack (with cross-attention to encoder output)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        """
        Project decoder output to vocabulary space.
        
        Args:
            x: Decoder output (batch, seq_len, d_model)
        
        Returns:
            Logits over vocabulary (batch, seq_len, vocab_size)
        """
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    """
    Build a complete Transformer model with the architecture from the paper.
    
    Default hyperparameters are from Table 3 of the paper (base model):
    - d_model = 512: Dimension of embeddings and model
    - N = 6: Number of encoder and decoder layers
    - h = 8: Number of attention heads
    - d_ff = 2048: Dimension of feed-forward network inner layer
    - dropout = 0.1: Dropout rate
    
    The paper also describes a "big" model with:
    - d_model = 1024, N = 6, h = 16, d_ff = 4096
    
    Args:
        src_vocab_size: Size of source vocabulary
        tgt_vocab_size: Size of target vocabulary
        src_seq_len: Maximum source sequence length
        tgt_seq_len: Maximum target sequence length
        d_model: Dimension of embeddings and model (default: 512)
        N: Number of encoder/decoder layers (default: 6)
        h: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        d_ff: Dimension of feed-forward inner layer (default: 2048)
    
    Returns:
        Complete Transformer model ready for training
    """
    # Create the embedding layers
    # Separate embeddings for source and target languages (can have different vocabularies)
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    # Separate positional encodings for source and target (can have different max lengths)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    # Stack of N identical encoder blocks
    encoder_blocks = []
    for _ in range(N):
        # Each encoder block has its own self-attention and feed-forward layers
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    # Stack of N identical decoder blocks
    decoder_blocks = []
    for _ in range(N):
        # Each decoder block has its own self-attention, cross-attention, and feed-forward layers
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    # Projects decoder output to target vocabulary size
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the complete transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters using Xavier uniform initialization
    # Xavier initialization helps with training stability by keeping signal variance consistent
    # The paper states: "We used the Adam optimizer... with β₁=0.9, β₂=0.98, and ε=10⁻⁹"
    # For weight initialization, Xavier/Glorot uniform is commonly used for Transformers
    for p in transformer.parameters():
        if p.dim() > 1:  # Only apply to weight matrices, not biases
            nn.init.xavier_uniform_(p)
    
    return transformer