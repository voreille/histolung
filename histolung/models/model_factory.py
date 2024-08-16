import yaml
from .unified_mil_model import UnifiedMILModel


def create_model(config_file: str):
    """
    Factory function to create a unified MIL model based on a YAML configuration file.
    
    Args:
        config_file (str): Path to the YAML configuration file.
        
    Returns:
        UnifiedMILModel: An instance of the unified MIL model.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Get parameters for the feature extractor and aggregation model from config
    feature_extractor_type = config['model']['type']
    feature_extractor_params = config['model']['params']
    aggregation_model_params = config['aggregation']

    # Create and return the unified MIL model
    unified_model = UnifiedMILModel(
        feature_extractor_type=feature_extractor_type,
        feature_extractor_params=feature_extractor_params,
        aggregation_model_params=aggregation_model_params)

    return unified_model
