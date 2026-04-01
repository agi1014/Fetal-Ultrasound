# Fetal Ultrasound Image Classification

An advanced, end-to-end deep learning pipeline built to classify fetal ultrasound images across six distinct planes, focusing on real-world medical imaging challenges such as dataset imbalance, model interpretability, and architectural comparison.

## Project Overview
The objective of this project is to accurately determine the anatomical plane of fetal ultrasound images. We utilized the publicly available **Fetal Planes DB (Zenodo)**, aiming to achieve high generalization leveraging modern Convolutional Neural Networks (CNNs). 

This repository documents the entire process: starting from a high-accuracy baseline using **EfficientNet-B3**, and introducing a strict structural comparison against **MobileNetV3-Large** to evaluate tradeoffs between parameter efficiency, speed, and accuracy on a restricted edge-like compute budget.

---

## Dataset & Preprocessing 
The dataset features extreme class imbalance across its six targets. To combat this, several techniques were implemented:
*   **Stratified Splitting**: Guaranteeing the exact label distributions across Training, Validation, and Held-Out Test sets.
*   **Weighted Random Sampling**: Over-sampling the minority classes (like `Fetal femur` and `Fetal thorax`) dynamically during the DataLoader loops to prevent the CNN from collapsing into a majority-class predictor.
*   **Heavy Augmentation Pipeline**: `RandomRotation`, `ColorJitter`, `RandomCrop`, and Multi-Axis Flipping to build resilience against ultrasound noise, artifact variance, and probe orientation.

---

## Architectural Implementations

### Model 1: EfficientNet-B3 (Baseline)
The baseline model leverages the Compound Scaling architecture of EfficientNet-B3.
*   **Input**: Normalized `300x300` resolution.
*   **Classifier**: Replaced the standard block with a custom dense structure: `Dropout(0.4) -> Linear(512) -> SiLU -> Dropout(0.2) -> Linear(6)`.
*   **Training Strategy**: 
    1.  **Phase 1 (Head-Tuning)**: Backbone structurally frozen; learning entirely on the dense classifier block with Cosine Annealing.
    2.  **Phase 2 (Progressive Unfreezing)**: Selected Deep Convolutional Blocks unfrozen at Epoch 8 with a 10x learning rate reduction to refine high-level fetal features without breaking ImageNet pre-training context.

### Model 2: MobileNetV3-Large (Comparison)
A lightweight architectural alternative optimized strictly for speed and deployment.
*   **Input**: Normalized native `224x224` resolution.
*   **Structure Alignment**: Stripped the internal default Sequential classifier and substituted it precisely with the Base Model's exact `SiLU/Dropout` classifier structure for a completely apples-to-apples assignment comparison.
*   **Unfreezing Strategy**: Only the final three Inverted Residual Blocks inside `backbone.features` were unlocked.

---

## Evaluation & Interpretability (Outputs)
Both notebooks automatically process and export detailed visual and statistical proofs into their respective `_outputs` directories:

### 1. Classification Metrics
Rather than relying purely on global accuracy, models are evaluated strictly on their `macro F1-scores` via precision-recall matrix tracking.

### 2. Saliency Maps (Grad-CAM)
Because medical imaging classifications require robust interpretability, both pipelines dynamically inject backward hooks into their concluding Conv Blocks (e.g. `layer4[-1]` or `features[-1]`). 
*   **Outputs**: Generates Class Activation Maps overlaying original ultrasounds to visually prove where the CNN is "looking" when isolating anatomic markers (e.g. the brain ridge vs the femur shaft).
*   **Function**: Confirms that the model isn't memorizing irrelevant noise or artifacts.

### 3. Row-Normalized Confusion Matrices
Used to highlight exact misclassification leakage patterns between confusing boundaries (like Maternal Cervix vs Other).

---

## Usage
These scripts are optimized to be dropped specifically into **Google Colab (T4 GPU)**.
1. Upload notebooks to Colab.
2. Select the T4 Compute Engine.
3. Automatically downloads data internally, extracts, trains, predicts, and auto-zips the outputs back to your browser.
