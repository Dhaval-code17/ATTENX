import os
import glob
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TestSample:
    image_path: str
    condition: str
    image_name: str
    expected_identity: Optional[str] = None

class DatasetManager:
    """
    Manages loading of evaluation dataset.
    Assumes structure:
    evaluation_dataset/
        condition_name/
            PersonName_*.jpg  (or just filename if identity unknown/irrelevant)
    """
    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root
        self.conditions = [
            d for d in os.listdir(dataset_root) 
            if os.path.isdir(os.path.join(dataset_root, d))
        ]

    def load_dataset(self) -> List[TestSample]:
        samples = []
        for condition in self.conditions:
            condition_path = os.path.join(self.dataset_root, condition)
            # Support common image extensions
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(condition_path, ext)))
            
            for img_path in image_files:
                basename = os.path.basename(img_path)
                # Heuristic: try to extract name from filename 
                # Expecting format: Name_AnySuffix.jpg or Name.jpg
                # If Condition is unknown_faces, expected_identity is None (or "Unknown")
                
                name_part = os.path.splitext(basename)[0].split('_')[0]
                
                if condition == "unknown_faces":
                    expected_identity = None # Should not match anyone in DB
                else:
                    expected_identity = name_part
                
                samples.append(TestSample(
                    image_path=img_path,
                    condition=condition,
                    image_name=basename,
                    expected_identity=expected_identity
                ))
        
        return samples

    def get_conditions(self) -> List[str]:
        return self.conditions
