#trainer.py

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import yaml  # Import yaml for saving config files
from tqdm import tqdm
from torchsummary import summary
from pytorch_forecasting.optim import Ranger

from src.models.unet import Unet, UNetConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UNetTrainer:
    """Trainer class for UNet model."""

    def __init__(
        self,
        model: Unet,
        config: UNetConfig,
        train_loader,  # Type hint cannot be specified without importing dataset class
        val_loader,    # Type hint cannot be specified without importing dataset class
        checkpoint_dir: str = "checkpoints",
        plot_dir: str = "plots",
    ) -> None:
        """
        Initialize UNetTrainer.

        Args:
            model (Unet): UNet model to train.
            config (UNetConfig): Configuration for training.
            train_loader: DataLoader for training dataset.
            val_loader: DataLoader for validation dataset.
            checkpoint_dir (str): Directory to save checkpoints. Defaults to "checkpoints".
            plot_dir (str): Directory to save plots. Defaults to "plots".
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(exist_ok=True)

        self.criterion = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss includes sigmoid
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        #Ranger(model.parameters(), lr=config.learning_rate)
        self.device = torch.device(config["device"])
        self.model.to(self.device)
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_optimizer_state = None
        self.best_epoch = 0
        self.best_train_loss = float('inf')


    def train_epoch(self, epoch: int) -> tuple[float, list[float]]:
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            tuple[float, list[float]]: Average training loss and list of batch losses for the epoch.
        """
        self.model.train()
        total_loss = 0
        batch_losses = []

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch} Training', leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            targets = batch['mask'].to(self.device).squeeze(1)
            self.optimizer.zero_grad()
            output = self.model(images).squeeze(1)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_losses.append(loss.item())

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, batch_losses

    def validate(self, epoch: int) -> tuple[float, list[float]]:
        """
        Validate the model for one epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            tuple[float, list[float]]: Average validation loss and list of batch losses for the epoch.
        """
        self.model.eval()
        total_loss = 0
        batch_losses = []

        progress_bar = tqdm(self.val_loader, desc=f'Epoch {epoch} Validating', leave=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                images = batch["image"].to(self.device)
                targets = batch["mask"].to(self.device).squeeze(1)

                output = self.model(images).squeeze(1)

                loss = self.criterion(output, targets)
                total_loss += loss.item()
                batch_losses.append(loss.item())
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, batch_losses

    def plot_learning_curves(self, train_losses: list[float], val_losses: list[float], save_path: Path) -> None:
        """
        Plot training and validation loss curves.

        Args:
            train_losses (list[float]): List of training losses per epoch.
            val_losses (list[float]): List of validation losses per epoch.
            save_path (Path): Path to save the plot.
        """
        sns.set_theme(style="darkgrid")

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        sns.lineplot(x=epochs, y=train_losses, label='Train Loss', linewidth=2, color="#3498db")
        sns.lineplot(x=epochs, y=val_losses, label='Validation Loss', linewidth=2, color="#e74c3c")

        plt.title('Training and Validation Loss Over Epochs', fontsize=16, fontweight='bold', color="#2c3e50")
        plt.xlabel('Epochs', fontsize=14, color="#34495e")
        plt.ylabel('Loss', fontsize=14, color="#34495e")
        plt.legend(fontsize=12, facecolor='white', edgecolor='lightgray')
        plt.xticks(epochs)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Learning curves plot saved to: {save_path}")


    def train(self, plot_curves: bool = True) -> None:
        """
        Full training loop.

        Args:
            plot_curves (bool): Whether to plot learning curves after training. Defaults to True.
        """
        train_losses = []
        val_losses = []
        
        #plot the model summary before training
        summary(self.model, input_size=tuple(self.config["input_size"]), device=self.config["device"], batch_size=self.config["batch_size"])

        start_datetime_str = datetime.now().strftime("%d-%m-%Y_%H%M") # for folder name

        epoch_progress = tqdm(range(1, self.config["epochs"] + 1), desc="Epochs", leave=True)
        for epoch in epoch_progress:
            logger.info(f"Starting Epoch {epoch}/{self.config["epochs"]}")

            train_loss, train_batch_losses = self.train_epoch(epoch)
            val_loss, val_batch_losses = self.validate(epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            epoch_progress.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"})

            logger.info(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                self.best_optimizer_state = self.optimizer.state_dict()
                self.best_epoch = epoch
                self.best_train_loss = train_loss
                logger.info(f"Validation loss improved. Best checkpoint updated at epoch {epoch}.")
            else:
                logger.info(f"Validation loss did not improve.")

        # After training, save the best checkpoint in a dated subfolder
        checkpoint_subfolder = self.checkpoint_dir / start_datetime_str
        checkpoint_subfolder.mkdir(exist_ok=True)
        checkpoint_filename_base = f"unet_checkpoint_best_{datetime.now().strftime('%d-%m-%Y_%H%M')}"
        checkpoint_filepath = checkpoint_subfolder / f"{checkpoint_filename_base}.pt"
        config_filepath = checkpoint_subfolder / f"{checkpoint_filename_base}.yaml"

        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.best_model_state,
            'optimizer_state_dict': self.best_optimizer_state,
            'train_loss': self.best_train_loss,
            'val_loss': self.best_val_loss
        }, checkpoint_filepath)
        logger.info(f"Best checkpoint saved to: {checkpoint_filepath}")

        # Save configuration to yaml file
        config_to_save = {
            'config': self.config,
            'metrics': {
                'epoch': self.best_epoch,
                'train_loss': self.best_train_loss,
                'val_loss': self.best_val_loss
            }
        }
        with open(config_filepath, 'w') as yaml_file:
            yaml.dump(config_to_save, yaml_file, indent=4)
        logger.info(f"Configuration saved to: {config_filepath}")


        if plot_curves:
            plot_filename = f"learning_curves_{start_datetime_str}.png"
            plot_filepath = self.plot_dir / plot_filename
            self.plot_learning_curves(train_losses, val_losses, plot_filepath)