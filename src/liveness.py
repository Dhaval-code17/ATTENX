
import numpy as np
import cv2

def is_live(face1, face2, threshold=1.5):
    """
    Checks if the subject is live based on facial landmark movement.
    
    Args:
        face1: InsightFace face object from frame 1
        face2: InsightFace face object from frame 2
        threshold: Minimum average pixel movement required. Default: 1.5.
    
    Returns:
        bool: True if live (movement > threshold), False otherwise.
    """
    if face1 is None or face2 is None:
        return False

    # Extract landmarks (shape: (5, 2))
    kps1 = face1.kps
    kps2 = face2.kps
    
    # Calculate Euclidean distance for each landmark pair
    distances = np.linalg.norm(kps1 - kps2, axis=1)
    
    # Average movement across all 5 landmarks
    avg_movement = np.mean(distances)
    
    # Debug info (optional, but requested to be explainable)
    # print(f"[DEBUG] Average Landmark Movement: {avg_movement:.2f} pixels")
    
    # Logic: Real faces have micro-movements (breathing, posture adjustments)
    # Static photos held perfectly still have near-zero movement (only noise)
    return avg_movement > threshold
