# AI vs Real Image Classifier

**Python | PyTorch | CNN |**
**July 2025**

This project implements a binary image classifier to distinguish **AI-generated images** from **real photographs** using a dataset of 120,000 labeled samples from [CIFAKE](https://www.kaggle.com/datasets). The repository includes both training and inference scripts, as well as visual results.

---

## Model Details

- **Architecture:** 5-layer CNN (Conv → ReLU → MaxPool)
- **Input:** Images resized to 224×224
- **Optimizer:** Adam, LR=0.001
- **Loss Function:** Binary Cross-Entropy
- **Data Augmentation:** Random flips, rotations, normalization
- **Early Stopping:** Patience = 5 epochs

**Performance:**

- CNN Accuracy on Test Set: 96.2%
- F1 Score: ~0.965
- Final Training Loss: 0.0837
- Final Validation Loss: 0.1161

---

## Results

### 1. Training Curves

![Training Curves](results/training_curves.png)

Shows the **train/validation loss** and **accuracy** over epochs.

### 2. Sample Predictions

![Sample Predictions](results/sample_predictions.png)

Displays **balanced predictions** of FAKE and REAL images with predicted confidence scores.

### 3. Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

Provides a clear overview of **true vs predicted labels**.

---

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
2. Train the CNN (optional)
bash
Copy
Edit
python train.py
Saves the trained model in models/cnn_model.pth.

Generates training log and plots in results/.

3. Run Inference
Open inference.ipynb (or run as a Python script) to:

Generate predictions for test images

Display sample predictions

Plot confusion matrix

Save figures in results/ for GitHub viewing

🛠 Technologies
Python 3.12

PyTorch

torchvision

PIL / Pillow

matplotlib

seaborn

scikit-learn

📌 Notes
The data/ folder is not included due to size. Download the CIFAKE dataset from Kaggle.

The repository is designed for clarity and reproducibility of results. Figures and logs are saved automatically in results/.
```
