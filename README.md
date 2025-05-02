# Sentiment-Analysis
This project implements a cutting-edge unsupervised anomaly detection framework tailored for medical imaging, particularly retinal OCT scans, using a novel Heterogeneous Autoencoder (Hetero-AE) that combines CNN and Transformer architectures for enhanced detection accuracy and interpretability.

# Project Summary 
Detect anomalies in medical images (like tumors, lesions, or structural irregularities) without requiring labeled abnormal samples by training a model to learn the distribution of normal data and identify deviations.

Model Architecture:
Encoder: Pretrained ResNet18 CNN to extract local and hierarchical features.
Decoder: Hybrid CNN-Transformer with Multi-Scale Sparse Transformer Block (MSTB) for global context modeling and Efficient Channel Attention (ECA) for refined feature emphasis.

Key Innovations:
Combines CNN local feature learning + Transformer long-range dependency modeling.
Uses MSTB to reduce Transformer computational cost while retaining multi-scale detail.
Integrates ECA for lightweight, adaptive channel attention, boosting discriminative power.
Produces interpretable heatmaps and anomaly maps for clinical decision support.

# Tools & Technologies
| **Category**           | **Details**                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| Programming Language   | Python                                                                                             |
| Frameworks & Libraries | PyTorch, NumPy, Matplotlib, torchvision                                                            |
| Dataset                | Retinal Optical Coherence Tomography (OCT) dataset (Kermany et al., 2018)                          |
| Model Components       | ResNet18 (encoder), CNN layers, Transformer blocks, MSTB, ECA, hybrid decoders                     |
| Optimization           | Adam optimizer, learning rate scheduler, gradient clipping, combined feature + reconstruction loss |
| Evaluation Metrics     | AUC, precision, recall, accuracy, F1-score, anomaly maps, heatmaps                                 |

# Performance Highlights
| **Model Variant**             | **AUC** | **Precision** | **Recall** | **Accuracy** |
| ----------------------------- | ------- | ------------- | ---------- | ------------ |
| CNN + CNN (baseline)          | 0.73    | 0.82          | 0.90       | 0.70         |
| HeteroAE                      | 0.85    | 0.85          | 0.93       | 0.85         |
| HeteroAE + MSTB               | 0.96    | 0.92          | 0.95       | 0.91         |
| HeteroAE + MSTB + ECA (final) | 0.95    | 0.92          | 0.97       | 0.94         |

# Result
Improved Recall — Final model achieves 97% recall, significantly reducing false negatives.
High AUC — AUC reaches ~0.95–0.96, showing robust discrimination between normal and abnormal cases.
Enhanced Interpretability — Generated heatmaps accurately localize anomalous regions, providing visual cues for clinical review.
Efficient Computation — ECA integration improves attention without major parameter overhead, enabling near real-time inference on high-resolution images.

