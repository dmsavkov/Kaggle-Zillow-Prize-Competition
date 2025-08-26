from omegaconf import OmegaConf

from .paths import CONFIG_PATH

def load_config_no_wrap(file_path='default'):
    if file_path == 'default':
        file_path = CONFIG_PATH / 'config.yaml'
    
    return OmegaConf.load(file_path)

def create_config_from_dict(config_dict):
    return OmegaConf.create(config_dict)

def merge_configs(base_config, new_config):
    return OmegaConf.merge(base_config, new_config)

def save_config(config, file_path):
    if not file_path.endswith('.yaml'):
        raise ValueError("File path must end with .yaml extension.")
    
    OmegaConf.save(config, file_path)
    print(f"Configuration saved to {file_path}")
    return 1
