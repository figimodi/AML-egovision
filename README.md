# Multimodal Egocentric Action Recognition

## Abstract
Egocentric vision focuses on action recognition from a first-person perspective. This project investigates sampling techniques and multimodal data integration for improving egocentric action recognition. Using the Epic-Kitchens and ActionSense datasets, the project explores spatial and temporal information trade-offs in RGB frames and incorporates ElectroMyoGraphy (EMG) data for multimodal fusion. The research highlights the potential of combining modalities through mid-level fusion to enhance classification performance.

## Project Overview
### Motivation
Egocentric action recognition poses unique challenges due to the reliance on first-person video recordings. While RGB information is often sufficient, additional modalities like audio and sensor data (e.g., EMG) can provide complementary information. This project investigates:
- Sampling strategies for RGB frames.
- The integration of EMG data with RGB frames.
- Late and mid-level fusion techniques for multimodal action recognition.

### Datasets
1. **Epic-Kitchens**
   - First-person videos of unscripted actions in diverse kitchen environments.
   - Focus on RGB frame sampling (dense and uniform).
2. **ActionSense**
   - Multimodal data, including RGB video, EMG signals, and spectrograms.
   - Data recorded using wearable sensors in controlled environments.

### Sampling Techniques
- **Dense Sampling**: Focuses on spatial information by selecting adjacent frames.
- **Uniform Sampling**: Prioritizes temporal information by selecting evenly spaced frames.
- **Finding**: Dense sampling with 16 frames per clip outperformed uniform sampling, achieving the best trade-off for egocentric action recognition.

### Feature Extraction
- Extracted features using the **Inflated 3D Convolutional Networks (I3D)** pre-trained model.
- Visualized features using dimensionality reduction techniques like PCA and t-SNE.

## Multimodal Analysis
### EMG Data
- Preprocessed with filtering, scaling, and zero-padding.
- Modeled using single and double-layer LSTM networks.

### Spectrograms
- Generated from EMG data and processed with a **LeNet5** CNN model.

### RGB Data
- Processed using dense sampling and classified with a single-layer LSTM network.

### Fusion Techniques
1. **Late Fusion**: Combines pre-trained model outputs at inference time.
2. **Mid-Level Fusion**: Jointly trains models by combining mid-level features, enhancing performance.

## Experiments
### Epic-Kitchens Results
- Dense sampling (16 frames per clip) achieved the highest Top-1 accuracy of 60.23% using LSTM.

### ActionSense Results
- Individual modalities:
  - RGB: 78.57% Top-1 accuracy (LSTM).
  - EMG: 56.97% Top-1 accuracy (1-layer LSTM).
  - Spectrograms: 54.50% Top-1 accuracy (LeNet5).
- Fusion approaches:
  - Mid-level fusion (RGB+EMG): 80.92% Top-1 accuracy.
  - Late fusion struggled to surpass individual modality performances.

## Conclusion
The project demonstrates the potential of multimodal approaches in egocentric action recognition:
- Dense sampling of RGB frames provides superior spatial and temporal trade-offs.
- Mid-level fusion of RGB and EMG enhances classification performance.
