# Traffic Sensor Data Analysis and Clustering

This repository contains code and data for analyzing traffic sensor data, performing graph-based clustering, and reconstructing traffic patterns.

## 📊 Dataset Description

### Data Files
- **week_median_data**: Primary dataset for graph clustering, containing median traffic patterns for sensors
- **distancewith0.csv**: Distance matrix between traffic sensors (obtained via Google Maps API)
- **train_by_cluster**: Training data organized by clusters
- **clustering_results.csv**: Output from the clustering stage, used for reconstruction

### Data Processing Pipeline
1. **Raw Data Processing**
   - Raw traffic count data is collected from sensors over a five-year period
   - Data is aggregated at weekly intervals
   - Median sequences are extracted to ensure consistency and handle missing data

2. **Distance Matrix Generation**
   - Traveling distances between sensors are computed using Google Maps API
   - Results are stored in `distancewith0.csv`
   - This matrix is used to define the graph adjacency matrix for clustering

## 🔄 Model Architecture

### 1. Graph Clustering Module
- Groups traffic sensors based on geospatial correlation
- Utilizes the distance matrix for defining sensor relationships
- Outputs cluster assignments in `clustering_results.csv`

### 2. Reconstruction Model
- Takes clustered data as input
- Uses `train_by_cluster` dataset for training
- Aims to reconstruct and predict traffic patterns

## 🚀 Usage

### Training
```bash
# Instructions for training the model will go here
```

### Testing
```bash
# Instructions for testing the model will go here
```

## 📈 Results
[Results and evaluation metrics will be added here]

## 📝 Citation
If you use this code or data in your research, please cite:
```bibtex
# Citation information will go here
```