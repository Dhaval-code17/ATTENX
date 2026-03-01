import argparse
import os
import pickle
import cv2
import numpy as np

# Try to import InsightFace, raise error if missing as per requirements
try:
    from insightface.app import FaceAnalysis
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "InsightFace is not installed. Please install it using:\n"
        "pip install insightface onnxruntime\n"
    ) from e


class FaceRegistrar:
    """
    Handles the registration of new faces into the system.
    Detects a single face, generates an embedding, and stores it.
    """

    def __init__(self, model_path='models/insightface_models', embeddings_path='data/embeddings.pkl'):
        """
        Initialize the FaceRegistrar.

        Args:
            model_path (str): Path to the InsightFace models directory.
            embeddings_path (str): Path to the pickle file for storing embeddings.
        """
        self.embeddings_path = embeddings_path
        self.model_path = model_path
        
        # Initialize the FaceAnalysis app (InsightFace)
        # Using 'buffalo_l' as required, CPU execution provider
        self.app = FaceAnalysis(
            name='buffalo_l',
            root=model_path,
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def load_embeddings(self):
        """
        Load existing embeddings from the pickle file.
        Returns an empty dictionary if the file does not exist.
        """
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_embeddings(self, data):
        """
        Save the updated embeddings dictionary to the pickle file.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(data, f)

    def register_face(self, image_path, student_id, name):
        """
        Register a face from an image.

        Args:
            image_path (str): Path to the input image.
            student_id (str): Unique ID for the student.
            name (str): Name of the student.

        Raises:
            FileNotFoundError: If image doesn't exist.
            ValueError: If no face or multiple faces are detected.
        """
        # 1. Load Image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # 2. Detect Face
        faces = self.app.get(image)
        
        # 3. Validation Rules
        if len(faces) == 0:
            raise ValueError("No face detected in the image. Please provide an image with exactly one face.")
        
        if len(faces) > 1:
            raise ValueError(f"Multiple faces ({len(faces)}) detected. Please provide an image with exactly one face.")

        print("Face detected")

        # 4. Generate Embedding
        # InsightFace 'faces' object already contains the embedding (normed_embedding)
        # We use the first (and only) face
        face_embedding = faces[0].normed_embedding
        print("Embedding generated")

        # 5. Store Data
        current_data = self.load_embeddings()
        
        # Create student entry
        student_entry = {
            "student_id": student_id,
            "name": name,
            "embedding": face_embedding
        }
        
        # Update / Append
        current_data[student_id] = student_entry
        
        self.save_embeddings(current_data)
        print("Student registered successfully")


def main():
    parser = argparse.ArgumentParser(description="Face Registration Utility")
    parser.add_argument('--image', required=True, help='Path to input image containing a single face')
    parser.add_argument('--id', required=True, help='Unique Student ID')
    parser.add_argument('--name', required=True, help='Full Name of the Student')
    
    args = parser.parse_args()

    try:
        registrar = FaceRegistrar()
        registrar.register_face(args.image, args.id, args.name)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
