# dataset/vio_real.py
import cv2, glob, torch
import numpy as np
from torch.utils.data import Dataset
class VIOReal(Dataset):
    def __init__(self, seq_roots, img_size=(480,640)):
        self.frames = []
        for root in seq_roots:
            self.frames += sorted(glob.glob(f"{root}/**/*.png", recursive=True))
        self.h, self.w = img_size
    def __len__(self):   return len(self.frames)
    def __getitem__(self, idx):
        img = cv2.imread(self.frames[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).float().unsqueeze(0)/255.0          # 1×H×W
        return {'image': img, 'filename': self.frames[idx]}