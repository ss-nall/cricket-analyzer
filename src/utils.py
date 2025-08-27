import numpy as np

def prepare_sample(kps_array, max_frames=316):
    """
    Convert keypoints array to model input.
    kps_array: np.array of shape [T, 33, 3]
    Returns: np.array of shape [1, max_frames, 99]
    """
    T = kps_array.shape[0]
    sample = kps_array.reshape(T, -1)  # flatten each frame to [33*3=99]
    
    if T < max_frames:
        pad = np.zeros((max_frames - T, sample.shape[1]))
        sample = np.vstack([sample, pad])
    else:
        sample = sample[:max_frames]
        
    return np.expand_dims(sample, 0)  # add batch dimension

def prepare_sample_from_array(kps_array, max_frames=316):
    """
    Same as prepare_sample but explicitly named for arrays already loaded.
    """
    return prepare_sample(kps_array, max_frames)
