import torch
import numpy as np
import random

def insertsection(sequence):
    """
    Scale down multiple sections of data using random scales between 0.4 and 0.6
    Args:
        sequence: Input array
    """
    A = sequence.copy()
    win_num = 5  # Fixed window number
    inx = random.sample(range(0, len(A)-5), win_num)
    slices = random.choice([5, 10, 20])
    
    for idx in inx:
        for i in range(slices):
            scale = round(random.uniform(0.4, 0.6), 1)
            A[idx:idx+i+1] = [x * scale for x in A[idx:idx+i+1]]
    
    return A

def insertpoint(sequence):
    """
    Insert random scaled down points into the data
    Args:
        sequence: Input array
    """
    A = sequence.copy()
    win_num = random.randint(10, 30)
    inx = random.sample(range(0, len(A)-5), win_num)
    for idx in inx:
        scale = round(random.uniform(0, 0.4), 1)
        A[idx] = A[idx] * scale
    
    return A

def insert_zero(sequence):
    """
    Randomly set sections to zero using random window numbers and indices
    Args:
        sequence: Input array
    """
    A = sequence.copy()
    win_num = random.randint(1, 5)  # Random number of windows between 1 and 5
    inx = random.sample(range(0, len(A)-5), win_num)  # Random starting indices
    slices = random.choice([5, 10, 20])  # Random slice size
    
    for idx in inx:
        for i in range(slices):
            if idx + i < len(A):
                A[idx:idx+i+1] = 0
    
    return A

def introduce_error(sequence):
    """
    Introduce all three types of errors into the data
    Args:
        sequence: Input array
    """
    error_type = random.choice(['section', 'point', 'zero'])
    
    if error_type == 'section':
        return insertsection(sequence)
    elif error_type == 'point':
        return insertpoint(sequence)
    else:
        return insert_zero(sequence) 