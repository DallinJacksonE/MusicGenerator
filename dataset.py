import os
import glob
import torch
from torch.utils.data import Dataset


class MIDIDataset(Dataset):
    def __init__(self, tensor_dir):
        # Recursively find all .pt files in the target directory
        self.file_paths = glob.glob(os.path.join(tensor_dir, "*.pt"))
        if not self.file_paths:
            print(f"WARNING: No tensors found in {
                  tensor_dir}. Did you run preprocess.py?")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the tensor instantly from the NVMe drive
        sequence = torch.load(self.file_paths[idx], weights_only=True)

        x = sequence[:-1]
        y = sequence[1:]
        return x, y
