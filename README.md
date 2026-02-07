# Hierarchical Plankton Identification, Correction, and Biomass Estimation Pipeline
This repository contains a **machine learning based pipeline** for automated plankton identification from **microscopic imagery** and **biomass estimation**, with results visualized through a **real time Flask web interface**.
The system is designed for **marine and aquatic monitoring**, with future extensibility toward population dynamics analysis and early-warning systems.
## Overview
The pipeline performs the following steps:
1. Image-based plankton classification using a hierarchical CNN
2. Parallel prediction at order, family, and species levels
3. Temporal de-duplication of detections to remove redundant frames
4. Biomass estimation using taxon-specific carbon conversion factors
5. Live visualization of species trends and abundance via Flask
## Dataset Construction and Curation
### Class Balancing Strategy
To address severe class imbalance typical of plankton datasets:
- Very small classes (< 100 images) were removed
- Dominant classes were capped to prevent bias
- Underrepresented classes were oversampled up to ~5,000 samples
Oversampling was performed using **random augmentation** (via Pillow) during data loading, ensuring:
- No duplication of stored images
- Reduced risk of memorization
- Improved generalization
### Taxonomy Curation
No unified taxonomy descriptor was initially available. Taxonomic labels were curated manually by querying and cross-checking:
- **NCBI Taxonomy**
- **WoRMS**
- **ITIS**
- **Catalogue of Life**
Special handling included:
- Infraorders or missing families labeled as name_uk
- Cases such as Cladoceromorpha (infraorder, not family) handled explicitly
Due to taxonomy completeness and reliability, **family level classification** was prioritized initially, with future expansion to finer taxonomic levels.
## Dataset Representation
The dataset is managed using a **CSV based addressing system**:
- The CSV file stores image paths and labels
- Images are accessed directly from disk at runtime
- No symlinks or redundant copying are used
- Random augmentation is applied **each time an image is loaded**
# Model Architecture
## Hierarchical CNN (Multi-Task Learning)
A **single CNN** with a shared backbone is used, producing **three outputs in parallel**:
1. Order level classifier
2. Family level classifier
3. Species level classifier
### Key characteristics:
- One input image → three predictions simultaneously
- **Hierarchical multi-task loss**
- Faster inference than sequential pipelines
- Prevents error propagation between taxonomic levels
The model definition is isolated in def_model, with losses and optimizers defined separately and imported into training scripts.
## Training Strategy
- Training performed on GPU-enabled environments 
- Data loaded dynamically from CSV
- Validation loss monitored each epoch
## Evaluation
Model evaluation is performed using a dedicated evaluation script on a held-out test set, without data leakage.
### Final Test Set Performance
- Order Accuracy:   95.19% (19009 / 19969)
- Family Accuracy:  91.20% (18211 / 19969)
- Species Accuracy: 89.28% (17829 / 19969)
## Temporal Cleaning and Logging
- Each video frame is classified independently 
- Predictions are temporally cleaned using a short time threshold (~100 ms)
- Redundant detections are removed
- Cleaned results are:
- - Saved to CSV
- - Served to the Flask dashboard
- Dashboard updates:
- - Cleaned data every ~5 seconds
- - Time series plots every ~10 seconds
## Biomass Estimation
Species specific **carbon conversion factors (µg C per individual)** are applied to cleaned species counts to compute:
- Biomass (mg C)  
This enables ecological interpretation beyond simple classification.
## Deployment Notes
- Final trained models are in .pth, .pkl 
- 64-bit OS is mandatory on the Raspberry Pi
- Designed for edge deployment using camera or video input
- Prototype currently uses recorded microscopy videos for stability
## Repository Structure

plankton-identification/\
├── main.py\
├── requirements.txt\
├── README.md\
│\
├── models/\
│   ├── *.pth\
│   └── *.pkl\
│\
├── csv_logs/\
│   ├── plankton_predictions_log.csv\
│   └── plankton_unique_log.csv\
│\
└── templates/\
    └── index.html

## Future Work
- Train fully multimodal image + flow-cytometry model
- Integrate real-time cytometry sensors
- Add population dynamics and warning systems
- Extend to additional datasets and taxonomic groups
- Improve robustness for long-term field deployment
## Author
Esther Christinal  
Undergraduate Bioengineering  
Focus: Computational Biology, Machine Learning

