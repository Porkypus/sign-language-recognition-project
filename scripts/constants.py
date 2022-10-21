import os
import numpy as np
import mediapipe as mp


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Path for exported data, numpy arrays
DATA_PATH = os.path.join(os.getcwd(), "data")

# Actions that we try to detect
actions = np.array(["hello", "thanks", "iloveyou"])

# Number of videos
no_sequences = 15

# Number of frames for each video
sequence_length = 30
