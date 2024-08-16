import torch
import torch.optim as optim

from histolung.models.model_factory import create_model
from histolung.mil.mil_trainer import MILTrainer
from histolung.mil.loss import MILLoss
from histolung.mil.metrics import accuracy
from histolung.utils.config_loader import load_config


def main():
    # Load configuration
    config_file = "config/model_config.yaml"
    model = create_model(config_file)

    # Set up criterion and optimizer
    criterion = MILLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create trainer
    trainer = MILTrainer(model,
                         criterion,
                         optimizer,
                         device="cuda" if torch.cuda.is_available() else "cpu")

    # Dummy training loop (replace with actual DataLoader)
    for epoch in range(10):
        # Assume bag_images and labels are obtained from a data loader
        loss = trainer.train_step(bag_images, labels)
        print(f"Epoch {epoch}, Loss: {loss}")

        # Validate every epoch (replace with actual validation DataLoader)
        val_metrics = trainer.validate(val_loader)
        print(
            f"Validation Loss: {val_metrics['val_loss']}, Accuracy: {val_metrics['accuracy']}"
        )


if __name__ == "__main__":
    main()
