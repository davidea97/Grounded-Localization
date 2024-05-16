import yaml
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def print_config(config):
    print("Configurations:")
    for key, value in config.items():
        print(f"{key}: {value}")
