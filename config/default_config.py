"""
Default configuration for DGCN model
"""

class Config:
    # Data parameters
    data_dir = "data/"
    train_cluster_dir = "data/train_by_cluster/"
    distance_matrix_file = "data/distancewith0.csv"
    clustering_result_file = "data/clustering_result_july13.csv"
    station_coordinates_file = "data/station_coordinates.csv"
    
    # Model dimensions
    time_dimension = 52      # h: input/output time dimension
    hidden_dimension1 = 64   # y: first hidden dimension
    hidden_dimension2 = 32   # z: second hidden dimension
    hidden_dimension3 = 16   # w: third hidden dimension
    order = 2               # k: diffusion steps
    
    # Training parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100
    corrupted_ratio = 0.3   # Ratio of batch samples to corrupt
    
    # Error injection parameters
    section_length = 4      # Length of section to corrupt
    num_points = 5         # Number of points to corrupt in point injection
    zero_ratio = 0.1       # Ratio of values to set to zero
    
    # Device configuration
    device = "cuda"  # or "cpu"
    
    # Logging
    checkpoint_dir = "checkpoints/"
    
    def update(self, **kwargs):
        """Update configuration parameters from kwargs"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}") 