import os
import numpy as np
import mediapipe as mp


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Path for exported data, numpy arrays
DATA_PATH = os.path.join(os.getcwd(), "np_data")
DATASET_PATH = "dataset/"

# Number of frames for each video
sequence_length = 50
