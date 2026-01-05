# Dataset preparation for Transformer training
# Handles bilingual parallel text data for sequence-to-sequence translation

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    """
    Dataset for bilingual (parallel) text data used in machine translation.
    
    Handles:
    - Tokenization of source and target sentences
    - Padding sequences to fixed length
    - Creating attention masks for padding and causal decoding
    - Adding special tokens (SOS, EOS, PAD)
    
    The dataset returns properly formatted inputs for the Transformer model:
    - encoder_input: Source sentence with SOS and EOS tokens
    - decoder_input: Target sentence with SOS token (for teacher forcing)
    - label: Target sentence with EOS token (ground truth for training)
    - encoder_mask: Prevents attention to padding in source
    - decoder_mask: Prevents attention to padding AND future tokens in target
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Special tokens for sequence boundaries and padding
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)  # Start of sequence
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)  # End of sequence
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)  # Padding

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Get source-target sentence pair
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenize: Convert text to token IDs
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate padding needed to reach seq_len
        # Encoder: Need space for SOS + tokens + EOS
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # Decoder input: Need space for SOS + tokens (EOS goes in label)
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Ensure sentences fit within seq_len
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Construct encoder input: [SOS] + tokens + [EOS] + [PAD]...
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Construct decoder input: [SOS] + tokens + [PAD]...
        # Used for teacher forcing during training
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Construct label: tokens + [EOS] + [PAD]...
        # Ground truth for training (shifted by one position from decoder_input)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Verify all sequences are exactly seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # Encoder mask: Allow attention to all non-padding tokens
            # Shape: (1, 1, seq_len) for broadcasting
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # Decoder mask: Prevent attention to padding AND future positions
            # Combines padding mask with causal mask (upper triangular)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,  # (seq_len)
            "src_text": src_text,  # Original source text for reference
            "tgt_text": tgt_text,  # Original target text for reference
        }
    
def causal_mask(size):
    """
    Create a causal (autoregressive) mask for the decoder.
    
    This mask prevents positions from attending to subsequent (future) positions,
    ensuring the autoregressive property during training.
    
    Creates an upper triangular matrix of 0s (positions to mask) with 1s on and
    below the diagonal (positions that can be attended to).
    
    Args:
        size: Sequence length
    
    Returns:
        Boolean mask of shape (1, size, size)
        - True: can attend
        - False: cannot attend (future position)
    
    Example for size=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    
    Position i can only attend to positions 0 to i (inclusive).
    """
    # torch.triu with diagonal=1 creates upper triangular matrix excluding diagonal
    # We negate it (== 0) to get lower triangular + diagonal
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0