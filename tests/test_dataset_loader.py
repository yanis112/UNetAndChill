import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import albumentations as A
import cv2
from typing import Dict, List, Tuple
from dotenv import load_dotenv 
import shutil
from src.data.dataset import MedicalSegmentationDataset, get_data_loaders  # Ajustez l'import selon votre structure

load_dotenv()  # Load environment variables


import numpy as np
from PIL import Image
import pandas as pd
import pytest

@pytest.fixture
def setup_test_data(tmp_path):
    """Create test data structure with dummy images."""
    data_dir = tmp_path / "data"
    splits_dir = data_dir / "splits"
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    
    # Create directories
    for d in [splits_dir, images_dir, masks_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Create dummy image and mask
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    
    # Save dummy files
    img_path = images_dir / "45.jpg"
    mask_path = masks_dir / "45_mask.png"
    Image.fromarray(dummy_image).save(img_path)
    Image.fromarray(dummy_mask).save(mask_path)
    
    # Create CSV files
    splits_data = pd.DataFrame({
        'image_filename': ['45.jpg'],
        'mask_filename': ['45_mask.png']
    })
    
    for split in ['train', 'val', 'test']:
        splits_data.to_csv(splits_dir / f"{split}.csv", index=False)
    
    return data_dir, splits_dir

@pytest.fixture
def sample_config():
    return {
        "batch_size": 2,
        "shuffle": True,
        "num_workers": 0  # Utiliser 0 pour les tests
    }

@pytest.fixture
def sample_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_mosaic(images: List[torch.Tensor], masks: List[torch.Tensor], save_path: Path, num_cols: int = 2):
    """
    Crée et sauvegarde une mosaïque d'images et leurs masques correspondants.
    """
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows * 2, num_cols, figsize=(num_cols * 4, num_rows * 6))
    if num_rows * num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1 or num_cols == 1:
        axes = axes.reshape(-1, 1) if num_cols == 1 else axes.reshape(1, -1)
    
    for idx in range(num_images):
        row = (idx // num_cols) * 2
        col = idx % num_cols
        
        # Dénormaliser l'image pour la visualisation
        img = images[idx].numpy().transpose(1, 2, 0)
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        # Afficher l'image
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Image {idx}')
        
        # Afficher le masque
        axes[row + 1, col].imshow(masks[idx].numpy(), cmap='gray')
        axes[row + 1, col].axis('off')
        axes[row + 1, col].set_title(f'Mask {idx}')
    
    plt.tight_layout()
    plt.savefig(save_path / 'batch_visualization.png')
    plt.close()

def test_dataset_initialization(setup_test_data):
    """Test l'initialisation correcte du dataset."""
    data_dir, splits_dir = setup_test_data
    
    dataset = MedicalSegmentationDataset(
        data_csv=str(splits_dir / "train.csv"),
        data_dir=data_dir,
        transform=None,
        phase="train"
    )
    
    assert len(dataset) > 0

def test_dataset_getitem(setup_test_data, sample_transform):
    """Test le fonctionnement de __getitem__."""
    data_dir, splits_dir = setup_test_data
    
    dataset = MedicalSegmentationDataset(
        data_csv=str(splits_dir / "train.csv"),
        data_dir=data_dir,
        transform=sample_transform,
        phase="train"
    )
    
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "image" in sample
    assert "mask" in sample
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    assert len(sample["image"].shape) == 3  # (C, H, W)
    assert len(sample["mask"].shape) == 2  # (H, W)

def test_dataloader(setup_test_data, sample_config, sample_transform, tmp_path):
    """Test le fonctionnement du dataloader et visualise un batch."""
    data_dir, splits_dir = setup_test_data
    
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        transform=sample_transform,
        config=sample_config
    )
    
    # Vérifier les propriétés de base du loader
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    
    # Obtenir et visualiser un batch
    batch = next(iter(train_loader))
    assert isinstance(batch, dict)
    assert "image" in batch
    assert "mask" in batch
    
    # Créer la visualisation
    create_mosaic(
        images=batch["image"],
        masks=batch["mask"],
        save_path=tmp_path,
        num_cols=2
    )
    
    # Vérifier que la visualisation a été créée
    assert (tmp_path / "batch_visualization.png").exists()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])