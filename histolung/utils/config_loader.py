import yaml
import os

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str):
        """
        Loads a YAML configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: A dictionary containing the loaded configuration.

        Raises:
            FileNotFoundError: If the specified configuration file does not exist.
            yaml.YAMLError: If the YAML file cannot be parsed.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                raise yaml.YAMLError(f"Error parsing YAML file: {exc}")

        # Optionally validate the configuration here
        ConfigLoader.validate_config(config)
        
        return config

    @staticmethod
    def validate_config(config: dict):
        """
        Validates the loaded configuration. This function checks if the necessary keys are present
        and have valid values. You can customize this based on your project's needs.

        Args:
            config (dict): The configuration dictionary.

        Raises:
            ValueError: If the configuration is invalid or missing critical sections.
        """
        required_keys = ['model', 'aggregation']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration section: '{key}'")

        # Example validation for model configuration
        if 'type' not in config['model']:
            raise ValueError("The 'model' section must contain a 'type' key.")
        
        if 'params' not in config['model']:
            raise ValueError("The 'model' section must contain a 'params' key.")
        
        if 'input_dim' not in config['aggregation'] or 'output_dim' not in config['aggregation']:
            raise ValueError("The 'aggregation' section must contain 'input_dim' and 'output_dim' keys.")

    @staticmethod
    def get_model_params(config: dict):
        """
        Helper function to extract model parameters from the configuration.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            dict: A dictionary of model parameters.
        """
        return config.get('model', {}).get('params', {})

    @staticmethod
    def get_aggregation_params(config: dict):
        """
        Helper function to extract aggregation parameters from the configuration.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            dict: A dictionary of aggregation parameters.
        """
        return config.get('aggregation', {})
