# main.py

import yaml

from src.data.dataset import get_data_loaders
from src.models.unet import Unet
from src.trainer.trainer import UNetTrainer
from src.trainer.tester import UNetTester


def main():
    # Load configurati
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # load the dataloaders
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir="data/raw", config=config
    )

    # # Initialize model and trainer
    model = Unet(config)
    trainer = UNetTrainer(
        model=model,
        config=config,
        train_loader=train_loader,  # Assume these are defined elsewhere
        val_loader=val_loader,
    )

    # Train model
    trainer.train()
    
    #--- Testing Phase ---
    # checkpoint_path = r"checkpoints\25-01-2025_1825\unet_checkpoint_best_25-01-2025_1830.pt"  # Replace with your actual checkpoint path
    # tester = UNetTester(
    #     model=model, # Use the same model instance, assuming you want to test the trained model
    #     config=config,
    #     test_loader=test_loader,
    #     checkpoint_path=checkpoint_path, # Path to the checkpoint you want to test
    # )
    # tester.test()


if __name__ == "__main__":
    main()
