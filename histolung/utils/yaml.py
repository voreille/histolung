import os
import re

import yaml


def expand_env_vars(data):
    """Recursively replace environment variables in strings within dictionaries and lists.
    
    Raises an error if an environment variable is not resolved.
    """
    if isinstance(data, dict):
        return {k: expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [expand_env_vars(item) for item in data]
    elif isinstance(data, str):
        expanded_data = os.path.expandvars(data)
        # Check if any unresolved ${VAR} patterns remain after expansion
        if re.search(r"\$\{([^}^{]+)\}", expanded_data):
            raise ValueError(f"Unresolved environment variable in: '{data}'")
        return expanded_data
    else:
        return data


def load_yaml_with_env(file_path):
    """
    Load a YAML file and substitute environment variables in all string values.
    
    Args:
        file_path (str or Path): The path to the YAML file.
    
    Returns:
        dict: The loaded YAML content with environment variables substituted.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return expand_env_vars(data)
