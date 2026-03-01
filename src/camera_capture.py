import cv2
import os
import sys

# Try to import the existing pipeline logic
try:
    from pipeline import main as run_pipeline
except ImportError:
    print("Error: Could not import pipeline. Make sure you are running from the project root.")
    sys.exit(1)

def ensure_directory(path):
    """
    Ensures that the directory exists. If not, creates it.
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"Error creating directory {path}: {e}")
            sys.exit(1)

def main():
    """
    Main function to handle camera capture and pipeline invocation.
    """
    # 1. Open system webcam
    # Index 0 is usually the default webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device (webcam).")
        print("Please check if the camera is connected and not used by another application.")
        sys.exit(1)

    # 4. Image handling setup
    save_dir = "images"
    save_filename = "captured.jpg"
    ensure_directory(save_dir)
    save_path = os.path.join(save_dir, save_filename)

    print("Camera initialized successfully.")
    print("instructions:")
    print("  Press 'c' to capture image and run attendance pipeline.")
    print("  Press 'q' to quit.")

    try:
        while True:
            # Read frame-by-frame
            ret, frame = cap.read()

            # Handle frame read failure
            if not ret:
                print("Error: Failed to receive frame (stream end?). Exiting ...")
                break

            # 2. Show live preview window
            cv2.imshow('Face Attendance - Live Camera', frame)

            # Wait for key press (1ms delay)
            key = cv2.waitKey(1) & 0xFF

            # 3. Camera behavior
            if key == ord('c'):
                print("\n[Capture] capturing image...")
                
                # Save captured image to disk (overwrite on each capture)
                cv2.imwrite(save_path, frame)
                print(f"[Capture] Image saved to: {save_path}")
                
                # 5. Integration: Call existing pipeline logic
                print("[Pipeline] Running attendance pipeline...")
                try:
                    # Pass the captured image path to pipeline
                    run_pipeline(save_path)
                except Exception as e:
                    print(f"[Pipeline] Error during processing: {e}")
                
                print("[System] Ready for next capture.\n")

            elif key == ord('q'):
                print("\n[System] Quitting...")
                break
            

    except KeyboardInterrupt:
        print("\n[System] Interrupted by user.")
        
    finally:
        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()
        print("[System] Camera released and windows closed.")

if __name__ == "__main__":
    main()
