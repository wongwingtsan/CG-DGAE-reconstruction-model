import os
import pandas as pd
import numpy as np
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(train_dir, cluster_file):
    """
    Load initial data and cluster information
    """
    arr = os.listdir(train_dir)
    namelist = []
    for station in arr:
        station = station[:-4]
        namelist.append(station)
    
    cluster = pd.read_csv(cluster_file)
    cluster = cluster[cluster['stationname'].isin(namelist)].reset_index()
    cluster_num = len(set(cluster.cluster.values))
    namelist = list(cluster['stationname'].values)
    
    print('all has {} cluster numbers'.format(cluster_num))
    print('input stationname has', len(namelist))
    
    return cluster, namelist

def create_stationname(filename, namelist):
    stations = pd.read_csv(filename)['Y0']
    stations = list(stations)

    stationname = []
    crdinstation = []
    for item in namelist:
        if item[0:8] in stations:
            stationname.append(item)
            crdinstation.append(item[0:8])
        else:
            pass
    
    crdinstation = list(set(crdinstation))
    print('stationname has ', len(stationname))
    print('coordinate locations has ', len(crdinstation))

    return stationname, crdinstation

def create_source(stationname, foldername):
    X = []
    for station in stationname:
        values = []
        fileaddress = foldername + station + '.csv'
        station_df = pd.read_csv(fileaddress, low_memory=False)
        values += station_df['Volume'].values.tolist()
        X.append(values)
    
    return X

def split_data(X, split_ratio=1):
    # Initialize lists to store training and testing sets
    training_set_ = []
    testing_set_ = []

    # Iterate through each row in the original array
    for row in X:
        num_elements = len(row)
        split_point = int(num_elements * split_ratio)
        
        # Split the row's elements into training and testing
        training_elements = row[:split_point]
        testing_elements = row[split_point:]
        
        # Append the training and testing elements to the respective sets
        training_set_.append(training_elements)
        testing_set_.append(testing_elements)

    # Convert the lists of lists to NumPy arrays
    training_ = np.array(training_set_)
    testing_ = np.array(testing_set_)

    # Check the shapes of the training and testing sets
    print("Training Set Shape:", training_.shape)
    print("Testing Set Shape:", testing_.shape)
    
    return training_, testing_

def generateDF(path_distance, path_coordinates, namelist):
    df = pd.read_csv(path_distance)
    names1 = list(df['Y0'])
    df = df.drop(columns=['Y0'])
    df.columns = names1
    CO = pd.read_csv(path_coordinates)
    df['stationname'] = CO.ID
    df = df.set_index('stationname')
    df = df.loc[namelist]
    df = df[namelist]
    
    return df

def generate_adj(stationname, DF):
    matrix = pd.DataFrame(columns=stationname, index=stationname)

    for station in stationname:
        for STATION in stationname:
            matrix.loc[station, STATION] = DF.loc[station[0:8], STATION[0:8]]

    MAX = matrix.max()
    matrix = matrix/MAX
    matrix = np.exp(-matrix.astype(float))
    matrix = matrix.apply(pd.Series.nlargest, axis=1, n=493)
    matrix = matrix.fillna(0)
    matrix = matrix.values

    adj = np.zeros_like(matrix)
    for i in range(len(matrix)):
        ind = np.argsort(matrix[i])[::-1][0:6]
        adj[i][ind] = 1

    print('The adjacent matrix shape is', adj.shape)
    
    return adj, matrix

def get_M(matrix, k):
    M = np.zeros_like(matrix)
    M = torch.Tensor(M)
    matrix = torch.Tensor(matrix)
    values, index = torch.topk(matrix, k)
    function = torch.nn.Softmax(dim=1)
    values = function(values)
    for i in range(len(M)):
        ind = index[i]
        M[i][ind] = values[i]
    
    return M.to(device)

def load_and_process_data(config):
    """
    Main function to load and process all data
    """
    # Load initial data
    cluster, namelist = load_data(config.train_cluster_dir, config.clustering_result_file)
    
    # Create station names and coordinates
    stationname, crdinstation = create_stationname(config.distance_matrix_file, namelist)
    
    # Create source data
    X = create_source(stationname, config.train_cluster_dir)
    
    # Split data
    training_, testing_ = split_data(X)
    
    # Generate distance DataFrame and adjacency matrix
    DF = generateDF(
        path_distance=config.distance_matrix_file,
        path_coordinates=config.station_coordinates_file,
        namelist=crdinstation
    )
    
    adj, A = generate_adj(stationname, DF)
    
    # Generate M matrix
    M = get_M(A, 12)
    
    return {
        'cluster': cluster,
        'stationname': stationname,
        'training': training_,
        'testing': testing_,
        'adj': adj,
        'M': M
    } 