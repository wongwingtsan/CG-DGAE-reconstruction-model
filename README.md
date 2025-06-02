# CG-DGAE: Cluster-guided Denoising Graph Auto-Encoder

This  for traffic sensor data analysis. The model uses cluster information to guide the denoising procerepository implements a Cluster-guided Denoising Graph Auto-Encoder (CG-DGAE)ss in a graph auto-encoder architecture. The clustering module is not included in this repo but in the other repo.


## Model Architecture

The model consists of two main components:
- D_GCN (Diffusion Graph Convolution Network)
- DGCN (Denoising Graph Convolutional Network with cluster guidance)

The architecture uses skip connections and combines MSE and KL divergence losses for better reconstruction.

## Data Requirements

The following data files are required:
- Training data in CSV format in `data/train_by_cluster/`
- Distance matrix file: `data/distancewith0.csv`
- Clustering results: `data/clustering_result_july13.csv`
- Station coordinates: `data/station_coordinates.csv`

Each station's data should be in a separate CSV file with a 'Volume' column.

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in the required format and update paths in `config/default_config.py`

2. Run training:
```bash
python train.py
```

The model will:
- Load and process station data
- Create adjacency matrices
- Train the model with data corruption
- Save the best model to `checkpoints/`

## Configuration

Key parameters in `config/default_config.py`:
- Model dimensions (h, y, z, w, k)
- Training parameters (batch_size, learning_rate, num_epochs)
- Data corruption parameters
- File paths

## Project Structure

```
├── config/
│   └── default_config.py     # Configuration parameters
├── models/
│   └── cg_dgae.py           # Model implementation
├── utils/
│   ├── data_utils.py        # Data loading and processing
│   └── data_augmentation.py # Data corruption functions
└── train.py                 # Training script
```

## Dependencies

See requirements.txt for detailed dependencies. Main requirements:
- PyTorch
- NumPy
- Pandas
- SciPy 

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{CG-DGAE2024,
    title = {Cluster-guided denoising graph auto-encoder for enhanced traffic data imputation and fault detection},
    journal = {Expert Systems with Applications},
    authors = {Yongcan Huang, Hao Zhen, Jidong J. Yang}
    year = {2024},
    doi = {https://doi.org/10.1016/j.eswa.2024.123584}
}
``` 