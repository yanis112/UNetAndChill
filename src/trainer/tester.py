#tester.py

import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
import logging
from src.models.unet import UNetConfig, Unet  # Assuming unet.py is in src/models/
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UNetTester:
    """Tester class for UNet model."""
    def __init__(self, model: Unet, config: UNetConfig, test_loader, checkpoint_path: str,
                 test_results_dir: str = "test_results"):
        """
        Initialize the UNetTester.

        Args:
            model (Unet): The UNet model to be tested.
            config (UNetConfig): Configuration dictionary.
            test_loader (DataLoader): DataLoader for the test dataset.
            checkpoint_path (str): Path to the checkpoint file (.pt) to load the model from.
            test_results_dir (str, optional): Directory to save test results plots. Defaults to "test_results".
        """
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.checkpoint_path = Path(checkpoint_path)
        self.test_results_dir = Path(test_results_dir)
        self.test_results_dir.mkdir(exist_ok=True)

        self.device = torch.device(config["device"])
        self.model.to(self.device)
        self._load_checkpoint() # Load checkpoint at initialization
        logger.info(f"Model loaded from checkpoint: {checkpoint_path}")

    def _load_checkpoint(self):
        """Loads model state from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device) # Ensure checkpoint is loaded on correct device
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded successfully from {self.checkpoint_path}")

    def test(self):
        """Run testing and generate result plots."""
        self.model.eval()  # Set model to evaluation mode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        progress_bar = tqdm(self.test_loader, desc='Testing', leave=True) # Progress bar for testing
        with torch.no_grad():  # Disable gradient calculation during inference
            for batch_idx, batch in enumerate(progress_bar):
                images = batch['image'].to(self.device)
                targets = batch['mask'].to(self.device).squeeze(1)

                outputs = self.model(images).squeeze(1)
                #apply the sigmoid function to the output to get the probability (because we used BCEWithLogitsLoss and the Unet model does not include the sigmoid function)
                outputs = torch.sigmoid(outputs)
                
                #print("outputs:",outputs)
                #print("Output_shape:",outputs.shape)
                predicted_masks = (outputs > 0.5).float() # Threshold to get binary masks

                self._plot_and_save_results(images, predicted_masks, targets, batch_idx, timestamp)

    def _plot_and_save_results(self, images, predicted_masks, targets, batch_idx, timestamp):
        """Plots images, predicted masks, and ground truth masks and saves the plot."""
        num_samples_to_plot = min(len(images), 4) # Plot up to 4 samples per batch for visualization
        fig, axes = plt.subplots(num_samples_to_plot, 3, figsize=(10, 5 * num_samples_to_plot)) # Adjust figure size as needed
        sns.set_theme(style="white") # Set plot theme

        for i in range(num_samples_to_plot):
            # Convert tensors to numpy arrays and move to CPU if on GPU for plotting
            img = images[i].cpu().numpy().transpose(1, 2, 0) # Assuming CHW format, convert to HWC
            pred_mask = predicted_masks[i].cpu().numpy()
            gt_mask = targets[i].cpu().numpy()

            # Ensure image is in the correct range for display (0-1 or 0-255), adjust if necessary
            img = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img # Normalize if not already normalized

            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Image")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(pred_mask, cmap='gray') # Use grayscale colormap for masks
            axes[i, 1].set_title("Predicted Mask")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(gt_mask, cmap='gray') # Use grayscale colormap for masks
            axes[i, 2].set_title("Ground Truth Mask")
            axes[i, 2].axis('off')

        plt.tight_layout()
        plot_filename = f"test_results_batch_{batch_idx}_{timestamp}.png"
        plot_filepath = self.test_results_dir / plot_filename
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        logger.info(f"Saved test results plot to: {plot_filepath}")