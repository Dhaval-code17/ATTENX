
import cv2
import os
import sys
import time

# Try to import necessary modules from the project
try:
    from register_face import FaceRegistrar, FaceAnalysis
    from pipeline import main as run_pipeline
    from recognize import FaceRecognizer
    from utils import cosine_similarity
    import liveness
except ImportError:
    print("Error: Could not import project modules. Make sure you are running from the project root.")
    sys.exit(1)

# Global FaceAnalysis app to avoid reloading for every operation
_app = None

def get_face_app():
    global _app
    if _app is None:
        model_path = 'models/insightface_models'
        _app = FaceAnalysis(name='buffalo_l', root=model_path, providers=['CPUExecutionProvider'])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app

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

def capture_image_with_liveness(prompt="Capture Image"):
    """
    Captures TWO frames with a delay to perform liveness detection.
    
    Args:
        prompt (str): Text to display in the console instructions.
        
    Returns:
        str: Path to the captured image (frame 1), or None if cancelled or liveness failed.
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video device (webcam).")
        return None

    save_dir = "images"
    save_filename = "temp_capture.jpg"
    ensure_directory(save_dir)
    save_path = os.path.join(save_dir, save_filename)

    app = get_face_app()

    print(f"\n--- {prompt} ---")
    print("  Instructions:")
    print("  1. Position your face in the frame.")
    print("  2. When ready, press 'c'.")
    print("  3. Hold steady but move your head slightly (or blink) for 1 second.")
    print("  Press 'c' to start capture sequence.")
    print("  Press 'q' to cancel/return.")

    capture_success = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to receive frame.")
                break

            cv2.imshow('Camera Menu - Live Preview', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                print("\n[INFO] Capture sequence started...")
                print("[INFO] Capturing Frame 1...")
                frame1 = frame.copy()
                
                # Perform quick detection on Frame 1 just to check face presence
                faces1 = app.get(frame1)
                if len(faces1) != 1:
                    print(f"[ERROR] Found {len(faces1)} faces in Frame 1. Please ensure exactly one face is visible.")
                    break # Go back to preview loop logic? No, let's break to retry or fail. 
                          # Ideally, user should retry, but simplistic flow: fail and return None?
                          # Let's clean up and return None so user can try again from Main Menu.
                    cap.release()
                    cv2.destroyAllWindows()
                    return None

                print("[INFO] Waiting 1 second. Please move slightly...")
                time.sleep(1.0)
                
                # Capture Frame 2
                # We need to read from camera buffer again to clear old frames?
                # Actually, video capture buffer might have old frames.
                # Let's read a few frames to flush buffer
                for _ in range(5):
                    cap.read()
                
                ret2, frame2 = cap.read()
                if not ret2:
                    print("[ERROR] Failed to capture Frame 2.")
                    break
                
                print("[INFO] Capturing Frame 2...")
                
                # Detect face in Frame 2
                faces2 = app.get(frame2)
                if len(faces2) != 1:
                    print(f"[ERROR] Found {len(faces2)} faces in Frame 2. Liveness check failed.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return None

                # Liveness Check
                print("[INFO] Performing liveness check...")
                is_real = liveness.is_live(faces1[0], faces2[0])
                
                if is_real:
                    print("[INFO] Liveness confirmed.")
                    cv2.imwrite(save_path, frame1) # Save Frame 1
                    capture_success = True
                else:
                    print("[ERROR] Spoof attempt detected (Liveness check failed).")
                    print("        Make sure to move slightly or blink between captures.")
                
                break
            elif key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return save_path if capture_success else None


def register_mode():
    """
    Handles the face registration process with duplicate check and liveness.
    """
    print("\n[INFO] Starting Registration Mode...")
    
    # Use new liveness capture function
    image_path = capture_image_with_liveness("Register Face")
    
    if image_path:
        print("\n[INFO] Image captured. Checking for existing registration...")

        try:
             # --- DUPLICATE CHECK START ---
            # 1. Get embedding (Using existing app instance)
            app = get_face_app()
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read captured image.")

            faces = app.get(img)

            if len(faces) == 0:
                print("[ERROR] No face detected. Cannot register.")
                return # Should unlikely happen as we checked in capture_image_with_liveness
            if len(faces) > 1:
                print(f"[ERROR] Multiple faces ({len(faces)}) detected. ensure only one person is in frame.")
                return

            new_embedding = faces[0].normed_embedding

            # 4. Load existing embeddings
            embeddings_path = 'data/embeddings.pkl'
            # Initialize recognizer just for loading embeddings
            recognizer = FaceRecognizer(model_path='models/insightface_models', embeddings_path=embeddings_path)
            known_embeddings = recognizer.known_embeddings

            # 5. Compare
            is_duplicate = False
            threshold = 0.5  # default threshold
            
            for student_id, data in known_embeddings.items():
                score = cosine_similarity(new_embedding, data['embedding'])
                if score > threshold:
                    print(f"[ERROR] Student already registered as {data['name']} ({student_id}).")
                    is_duplicate = True
                    break
            
            if is_duplicate:
                print("[INFO] Student already registered. Skipping.")
                return
            # --- DUPLICATE CHECK END ---

            # Continue with registration if not duplicate
            student_id = input("Enter Student ID: ").strip()
            name = input("Enter Student Name: ").strip()
            
            if not student_id or not name:
                print("Error: Student ID and Name are required.")
                return

            print(f"[INFO] Registering face for {name} ({student_id})...")
            
            # Using existing FaceRegistrar logic
            registrar = FaceRegistrar()
            registrar.register_face(image_path, student_id, name)
            
            print("[INFO] Embedding stored successfully.")
            
        except ValueError as ve:
            print(f"Registration Failed: {ve}")
        except Exception as e:
            print(f"An error occurred during registration: {e}")
    else:
        print("[INFO] Registration cancelled or Liveness Failed.")


def attendance_mode():
    """
    Handles the attendance marking process with liveness.
    """
    print("\n[INFO] Starting Attendance Mode...")
    
    image_path = capture_image_with_liveness("Mark Attendance")
    
    if image_path:
        print("\n[INFO] Image captured. Processing...")
        print("[INFO] Attendance marked (if match found)...")
        
        try:
            # Call existing pipeline logic
            run_pipeline(image_path)
        except Exception as e:
            print(f"Error during attendance processing: {e}")
    else:
        print("[INFO] Attendance cancelled or Liveness Failed.")


def main_menu():
    """
    Displays the main menu and routes user choices.
    """
    while True:
        print("\n==============================")
        print("   Face Attendance System")
        print("==============================")
        print("1. Register Face")
        print("2. Mark Attendance")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            register_mode()
        elif choice == '2':
            attendance_mode()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
