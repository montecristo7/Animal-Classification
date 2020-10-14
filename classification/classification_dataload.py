import os
import pathlib

from skimage import io
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from constant import default_train_transform, default_val_transform, get_image_category


class ClassificationDataset(Dataset):
    def __init__(self, set_name, root_dir, val_size=0.15, random_state=1, exclude_category=('Exclude',),
                 target_category='species_binary'):

        self.root_dir = pathlib.Path(root_dir) / set_name
        species_category = get_image_category(target_category=target_category)

        raw_files = [file.split('.')[0].split('_') for file in os.listdir(root_dir) if file.endswith('.jpg')]
        self.image_files = [[species_category.get(image[-8], 'Exclude')] + image for image in raw_files]
        if exclude_category:
            self.image_files = [image for image in self.image_files if image[0] not in exclude_category]

        self.species = [spec[0] for spec in self.image_files]
        unique_species = sorted(list(set(self.species)))
        self.species_classes_map = {spec: unique_species.index(spec) for spec in unique_species}
        self.classes = list(self.species_classes_map.keys())

        self.train, self.val = train_test_split(self.image_files, test_size=val_size, random_state=random_state,
                                                stratify=self.species)

        if set_name == 'full':
            self.dataset = self.image_files
            self.transform_func = default_train_transform
        elif set_name == 'train':
            self.dataset = self.train
            self.transform_func = default_train_transform
        elif set_name == 'validation':
            self.dataset = self.val
            self.transform_func = default_val_transform
        elif set_name == 'test':
            self.dataset = self.image_files
            self.transform_func = default_val_transform
        else:
            raise ValueError('Unknown set_name: ' + str(set_name))

        self.data_set_species = [spec[0] for spec in self.dataset]
        self.cam_trap = [spec[3] if len(spec) == 12 else '_'.join(spec[3:5]) for spec in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = pathlib.Path(self.root_dir) / '{}.jpg'.format('_'.join(self.dataset[idx][1:]))
        img = io.imread(img_path)
        img = img if self.transform_func is None else self.transform_func(img)

        target = self.dataset[idx][0]
        return img, self.species_classes_map[target]
