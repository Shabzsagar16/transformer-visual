# Configuration settings for Transformer training
# Contains hyperparameters, paths, and training settings

from pathlib import Path

def get_config():
    """
    Get training configuration dictionary.
    
    Hyperparameters:
    - batch_size: Number of samples per training batch
    - num_epochs: Total training epochs
    - lr: Learning rate for optimizer (Adam)
    - seq_len: Maximum sequence length for source and target
    - d_model: Transformer model dimension (default 512 from paper)
    
    Data:
    - datasource: HuggingFace dataset name
    - lang_src/lang_tgt: Source and target language codes
    
    Paths:
    - model_folder: Directory for saving model checkpoints
    - tokenizer_file: Tokenizer file naming pattern
    - experiment_name: TensorBoard experiment directory
    """
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,  # Model dimension from paper
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    """Construct path to weights file for a specific epoch."""
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    """Find the most recent weights file in the weights folder for checkpoint resumption."""
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
