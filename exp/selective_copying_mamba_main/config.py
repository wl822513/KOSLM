# Configuration for training
training_config = {
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_steps": 40000,  # 400K
    "num_page": 10  # 为了缓解显存压力，一次生成和处理一页的数据
}

# Configuration for dataset
dataset_config = {
    "l_noise": 24,  # number of padding tokens
    "l_memorize": 8,  # number of tokens to memorize
    "n_tokens": 8,  # alphabet size
    "intensity": 0,  # 10 %  0表示选择性复制
    "lag": False,
    "variable": True,  # Randomly distribute memorization tokens throughout sequence instead of frontloading them
    "variable_length": False,  # Randomize number of tokens to memorize
    "one_hot": False,
    "reverse": False,
    "static": False,
}


# Configuration for Mamba model
class MambaConfig:
    d_model: int = 64
    n_layer: int = 2
    vocab_size: int = dataset_config['n_tokens']
    ssm_cfg: dict = {}
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 1
    tie_embeddings: bool = False
    koss: bool = True  # 是否启用koss