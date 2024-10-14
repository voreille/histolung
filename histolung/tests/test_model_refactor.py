import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from histolung.models.models_legacy import MILModel, PretrainedModelLoader
from histolung.utils import yaml_load


class TestMILModelRefactor(unittest.TestCase):

    def setUp(self):
        # Set paths and configurations for testing
        self.modeldir = Path(
            "/home/valentin/workspaces/histolung/models/MIL/f_MIL_res34v2_v2_rumc_best_cosine_v3"
        )
        self.checkpoint = torch.load(
            self.modeldir / "fold_0" / "checkpoint.pt",
            weights_only=False,
        )

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Set some default configurations (use real cfg in practice)
        self.cfg = self.read_cfg()
        self.hidden_space_len = self.cfg.model.hidden_space_len

        # Initialize both old and new models (this assumes the old model is still accessible)
        self.old_model = self.initialize_old_model(self.cfg,
                                                   self.hidden_space_len)
        self.new_model = self.initialize_new_model(self.cfg,
                                                   self.hidden_space_len)

        # Load state dict from the checkpoint with strict=False
        self.old_model.load_state_dict(self.checkpoint["model_state_dict"],
                                       strict=False)
        self.new_model.load_state_dict(self.checkpoint["model_state_dict"],
                                       strict=False)
        self.old_model.to(self.device)
        self.new_model.to(self.device)

        self.old_model.eval()
        self.new_model.eval()

    def read_cfg(self):
        return yaml_load(self.modeldir /
                         "config_f_MIL_res34v2_v2_rumc_best_cosine_v3.yml")

    def initialize_old_model(self, cfg, hidden_space_len):
        """Initialize the old version of MILModel."""
        # Import the old model or define it inline if it's not modular
        from histolung.legacy.heatmaps import MIL_model as OldMILModel  # Assuming this is still accessible
        from histolung.legacy.models import ModelOption
        model_loader = ModelOption(
            cfg.model.model_name,
            cfg.model.num_classes,
            cfg.model.freeze_weights,
            cfg.model.num_frozen_layers,
            cfg.model.dropout,
            cfg.model.embedding_bool,
            cfg.model.pool_algorithm,
        )
        return OldMILModel(model_loader, hidden_space_len, cfg)

    def initialize_new_model(self, cfg, hidden_space_len):
        """Initialize the refactored version of MILModel."""
        model_loader = PretrainedModelLoader(
            cfg.model.model_name, cfg.model.num_classes,
            cfg.model.freeze_weights, cfg.model.num_frozen_layers,
            cfg.model.dropout, cfg.model.embedding_bool,
            cfg.model.pool_algorithm)
        return MILModel(model_loader, hidden_space_len, cfg)

    def compare_models(self, model1, model2):
        """Helper function to compare two models' weights."""
        for (name1, param1), (name2,
                              param2) in zip(model1.state_dict().items(),
                                             model2.state_dict().items()):
            self.assertTrue(torch.allclose(param1, param2, atol=1e-6),
                            f"Mismatch found in layer: {name1} vs {name2}")

    def test_model_weights_equality(self):
        """Test if the weights of the old and refactored models are the same."""
        self.compare_models(self.old_model.net, self.new_model.net)

    def test_model_predictions(self):
        """Test if the predictions of the old and refactored models are the same."""
        # Create random input data
        input_data = torch.randn(
            1, 3, 224,
            224)  # Adjust the shape based on your model's expected input

        # Run predictions on both models
        old_output = self.old_model(input_data, input_data)[0]
        new_output = self.new_model(input_data)[0]

        # Check if the predictions are the same (or very close)
        self.assertTrue(
            torch.allclose(old_output, new_output, atol=1e-6),
            "Predictions from the old and new models do not match")


if __name__ == '__main__':
    unittest.main()
