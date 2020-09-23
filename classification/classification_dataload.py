import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset


class Classification_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None, random_sample=None):
        if random_sample:
            self.landmarks_frame = pd.read_csv(csv_file, header=None, delimiter='\t').sample(random_sample)
        else:
            self.landmarks_frame = pd.read_csv(csv_file, header=None, delimiter='\t')
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.root_dir + self.landmarks_frame.iloc[idx, 0].split('/')[-1] + '.png'

        img = io.imread(img_path)

        landmarks = self.landmarks_frame.iloc[idx, 1]
        landmarks = landmarks.strip("][").replace("'", '').replace('"', '').split(', ')
        landmarks = np.array(landmarks)
        landmarks = landmarks.astype('float')

        x, y, w, h = landmarks
        landmarks = np.array([x, y, x + w, y + h])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(landmarks, dtype=torch.float32).view((1, 4))
        # there is only one class
        labels = torch.ones((1,), dtype=torch.int64)
        masks = torch.ones(img.shape[:-1], dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.landmarks_frame)
