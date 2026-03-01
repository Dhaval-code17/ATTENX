import cv2
import numpy as np
import os
import glob

def add_motion_blur(image, kernel_size=15):
    # Create the motion blur kernel
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel_motion_blur /= kernel_size
    return cv2.filter2D(image, -1, kernel_motion_blur)

def simulate_low_light(image, factor=0.3):
    # Convert to HSV, reduce V channel, convert back
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def add_mask(image):
    # Simple heuristic mask (gray rectangle over mouth/nose)
    h, w = image.shape[:2]
    # Assume face is roughly centered or detected. 
    # For robust test, we just occlude the bottom half of the center area
    # Better: Use dlib/mediapipe, but we want zero deps for generation if possible.
    # Let's just draw a rectangle on the bottom center.
    mask_h = int(h * 0.4)
    mask_w = int(w * 0.6)
    start_x = int((w - mask_w) / 2)
    start_y = int(h - mask_h - (h * 0.05)) # Bit from bottom
    
    cv2.rectangle(image, (start_x, start_y), (start_x + mask_w, start_y + mask_h), (200, 200, 200), -1)
    return image

def main():
    root = "evaluation_dataset"
    normal_dir = os.path.join(root, "normal")
    
    if not os.path.exists(normal_dir):
        print("Normal directory not found.")
        return

    images = glob.glob(os.path.join(normal_dir, "*.jpg"))
    if not images:
        print("No images found in normal directory.")
        return

    print(f"Found {len(images)} source images. Generating conditions...")

    for img_path in images:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        # 1. Low Light
        low_light = simulate_low_light(img.copy())
        cv2.imwrite(os.path.join(root, "low_light", filename), low_light)

        # 2. Motion Blur
        blur = add_motion_blur(img.copy())
        cv2.imwrite(os.path.join(root, "motion_blur", filename), blur)

        # 3. Masked
        masked = add_mask(img.copy())
        cv2.imwrite(os.path.join(root, "masked", filename), masked)
        
        # 4. Side Pose (Simulated by simple perspective warp - crude but works for pipeline testing)
        # Real side pose needs 3D rotation, but we can't generate that from 2D easily without heavy ML
        # Skipping simulated side pose, sticking to what we can reliably fake.

    print("Generation complete.")

if __name__ == "__main__":
    main()
