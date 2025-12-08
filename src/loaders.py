from pathlib import Path
import json
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Tuple
from PIL import Image

def _is_valid_file(filepath: Path) -> bool:
    """Helper function to filter any hidden .txt files, or files that are in hidden folders (e.g. ".ipynb_checkpoints", 
    ".git", etc.)"""

    check = lambda part: not part.startswith('.') or part in {'.', '..'}
    return all(map(check, filepath.parts))

class WhalesBaseDataset(Dataset, ABC):
    def __init__(self, dataset_dir, transform=None):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.image_dir, self.label_dir = self._get_paths()

        # Validate directories
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f'Image directory not found: {self.image_dir}\n'
            )
        if not self.label_dir.exists():
            raise FileNotFoundError(
                f'Labels directory not found: {self.label_dir}\n'
            )
        
        # Load te images
        all_image_paths = [p for p in self.image_dir.rglob('*.png') if _is_valid_file(p)]
        self.image_paths = sorted(all_image_paths, key=lambda path: path.name)

        # Load the labels
        self.labels_data = self._parse_all_labels()

        for img_path in self.image_paths:
            if img_path.name not in self.labels_data:
                raise FileNotFoundError(
                    f'No labels found for image: {img_path}\n'
                )
           
    @abstractmethod
    def _get_paths(self):
        pass

    @abstractmethod
    def _parse_label(self, label_path: Path):
        pass

    def _parse_all_labels(self):
        label_paths = self.label_dir.rglob('*.json')

        labels_data = {}
        for path in label_paths:
            if _is_valid_file(path):

                # Update the dictionary with all the label info (cache)
                labels_data |= self._parse_label(path)

        return labels_data
    
    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.image_paths)
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        labels = self.labels_data[image_path.name]
        return image, labels

    
class LineLevelDataset(WhalesBaseDataset):
    def _get_paths(self) -> Tuple[Path, Path]:
        return self.dataset_dir / 'images' / 'lines', self.dataset_dir / 'labels' / 'line_level'
    
    def _parse_label(self, label_path):
        with open(label_path) as js:
            data = json.load(js)

        labels_info = {}
        for entry in data['line_level_info']:
            labels_info[entry['image_name']] = {
                'unit_intervals': entry['unit_intervals'],
                'unit_classes': entry['unit_classes']
            }

        return labels_info
    

class PageLevelDataset(WhalesBaseDataset):
    def _get_paths(self):
        return self.dataset_dir / 'images' / 'pages', self.dataset_dir / 'labels' / 'page_level'



