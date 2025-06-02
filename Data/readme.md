You can download data files from [Google Drive](https://drive.google.com/drive/folders/1uboZBZwdevcPBgOViv2EEZ6ibdcMJ2U_?usp=sharing). 

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

