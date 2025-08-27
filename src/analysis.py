import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def compare_shots(ref_kps, user_kps, max_frames=316):
    """
    Compare reference shot vs user shot and provide similarity percentage and specific feedback.
    """

    def flatten_and_interpolate(seq, n=max_frames):
        # Flatten keypoints and interpolate to fixed frame count
        seq_flat = seq.reshape(seq.shape[0], -1)
        old_idx = np.linspace(0, 1, seq_flat.shape[0])
        new_idx = np.linspace(0, 1, n)
        return np.array([np.interp(new_idx, old_idx, seq_flat[:, i]) for i in range(seq_flat.shape[1])]).T

    ref_seq = flatten_and_interpolate(ref_kps)
    user_seq = flatten_and_interpolate(user_kps)

    # DTW similarity
    dtw_distance, _ = fastdtw(ref_seq, user_seq, dist=euclidean)
    max_possible_dist = np.prod(ref_seq.shape)
    similarity = max(0, 100 * (1 - dtw_distance / max_possible_dist))

    feedback = []

    # --- Bat angle (shoulder to wrist) ---
    try:
        ref_bat_angle = np.arctan2(ref_seq[:, 15*3+1] - ref_seq[:, 13*3+1],
                                   ref_seq[:, 15*3] - ref_seq[:, 13*3])
        user_bat_angle = np.arctan2(user_seq[:, 15*3+1] - user_seq[:, 13*3+1],
                                    user_seq[:, 15*3] - user_seq[:, 13*3])
        bat_diff = np.mean(np.abs(ref_bat_angle - user_bat_angle))
        if bat_diff > 0.15:
            feedback.append("Bat swing plane deviates; keep alignment similar to reference.")
    except:
        feedback.append("Bat angle could not be analyzed.")

    # --- Head stability (nose) ---
    try:
        head_diff = np.mean(np.abs(ref_seq[:,0*3+1] - user_seq[:,0*3+1]))
        if head_diff > 0.03:
            feedback.append("Head moves too much; focus on keeping eyes on the ball.")
    except:
        feedback.append("Head position could not be analyzed.")

    # --- Front foot placement ---
    try:
        foot_diff = np.mean(np.abs(ref_seq[:,27*3+1] - user_seq[:,27*3+1]))
        if foot_diff > 0.03:
            feedback.append("Front foot placement off; ensure solid base at impact.")
    except:
        feedback.append("Front foot position could not be analyzed.")

    # --- Back foot placement ---
    try:
        back_foot_diff = np.mean(np.abs(ref_seq[:,28*3+1] - user_seq[:,28*3+1]))
        if back_foot_diff > 0.03:
            feedback.append("Back foot balance differs; distribute weight correctly.")
    except:
        feedback.append("Back foot position could not be analyzed.")

    # --- Follow-through / wrist rotation ---
    try:
        ref_wrist = ref_seq[-1, 15*3:16*3]
        user_wrist = user_seq[-1, 15*3:16*3]
        if np.linalg.norm(ref_wrist - user_wrist) > 0.05:
            feedback.append("Follow-through differs; rotate wrists smoothly after impact.")
    except:
        feedback.append("Follow-through could not be analyzed.")

    # --- Timing / coordination check (simple) ---
    try:
        timing_diff = np.mean(np.abs(np.diff(ref_seq[:,15*3+1]) - np.diff(user_seq[:,15*3+1])))
        if timing_diff > 0.02:
            feedback.append("Swing timing differs; try syncing shot rhythm with reference.")
    except:
        feedback.append("Timing could not be analyzed.")

    # Ensure at least 3 feedback points
    while len(feedback) < 3:
        feedback.append("Refine your shot to better match reference posture and movement.")

    return round(similarity,2), feedback[:5]  # return max 5 detailed points
