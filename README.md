# 🏥 UNet for Medical Image Segmentation 🧠

![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/your-username/your-repo)
![GitHub last commit](https://img.shields.io/github/last-commit/your-username/your-repo)
![GitHub repo size](https://img.shields.io/github/repo-size/your-username/your-repo)

This repository contains a PyTorch implementation of the UNet architecture for medical image segmentation tasks. It leverages the power of `uv` for efficient virtual environment management and includes comprehensive training and testing pipelines.

## 🌟 Features

-   **UNet Model**: A flexible UNet implementation with configurable depth and filter count. 🛠️
-   **Data Handling**: Efficient data loading and preprocessing with `albumentations` support. 🖼️
-   **Training Pipeline**: Robust training loop with validation, checkpointing, and learning curve plotting. 📈
-   **Testing Pipeline**: Evaluate model performance on test data and visualize segmentation results. 🧪
-   **Reproducibility**: Detailed logging, model checkpointing, and configuration saving ensure reproducibility. ♻️
-   **Fast Experimentation**: Easy experimentation with different model configurations and training parameters. ⚡

## 🚀 Getting Started

Follow these steps to set up the project and start training your UNet model.

### Prerequisites

-   Python 3.9+ 🐍
-   [uv](https://github.com/astral-sh/uv) for virtual environment management
-   PyTorch 2.0+ 🔥

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. **Create and activate a virtual environment using `uv`:**

    ```bash
    uv venv
    source .venv/bin/activate
    ```

3. **Install dependencies using `uv`:**

    ```bash
    uv pip install -r requirements.txt
    ```

### 📁 Dataset Preparation

-   Place your dataset in the `data/raw` directory. The dataset should be organized as follows:

```
data/raw/
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── masks/
    ├── image1.png
    ├── image2.png
    └── ...
```

-   Create a CSV file (`dataset.csv`) listing the filenames of images and corresponding masks. You can refer to the example `dataset.csv` provided in the repository. The CSV should have two columns: `image_filename` and `mask_filename`.

-   Generate training, validation, and testing splits using `dataset.py`:

    ```bash
    python src/data/dataset.py
    ```

    This will create `train.csv`, `val.csv`, and `test.csv` in the `data/splits` directory.

## 🤖 Training the Model

1. **Configure Training Parameters:**
    -   Modify the `configs/config.yaml` file to set model and training parameters.
    -   Adjust parameters like `input_size`, `initial_filters`, `depth`, `batch_size`, `learning_rate`, `epochs`, etc.

2. **Start Training:**

    ```bash
    python src/main.py
    ```

-   Training progress and logs will be displayed in the console.
-   Checkpoints will be saved in the `checkpoints` directory.
-   Learning curve plots will be saved in the `plots` directory.

## 🧪 Testing the Model

To evaluate the trained model on the test dataset:

1. **Ensure a Checkpoint Exists:**
    -   Make sure you have a trained model checkpoint in the `checkpoints` directory.

2. **Uncomment the Testing Phase in `main.py`**
    -   Uncomment the testing section at the end of `src/main.py`

3. **Specify Checkpoint Path**
    -   Change the `checkpoint_path` variable in `src/main.py` to the checkpoint you wish to test

4. **Run Testing:**
    -   Execute the following command:

    ```bash
    python src/main.py
    ```

-   Test results, including visualizations of input images, predicted masks, and ground truth masks, will be saved in the `test_results` directory.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
