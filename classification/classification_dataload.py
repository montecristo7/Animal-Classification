import json
import os
import pathlib

import pandas as pd
import torchvision
from skimage import io
from torch.utils.data import Dataset

from constant import default_train_transform, default_val_transform


class ClassificationDataset(Dataset):
    def __init__(self, set_name, image_root='image_new', json_root='./json',
                 exclude_category=('Exclude', 'Human', 'Unknown'),
                 target_category='binary', flip_image=False):
        """
        The class is used to feed image data into pytorch model, inherited from torch Dataset.
        Args:
            set_name: str
                train, test, or val
            image_root: str
                image root folder
            exclude_category: tuple
                categories to be excluded
            target_category: str
                possible classified categories:
                    'species', 'binary', 'genus', 'family', 'order', 'class'
            flip_image: bool
                flag for train set only, if True rotate the image
        """
        self.image_root = pathlib.Path(image_root)

        json_root = pathlib.Path(json_root)
        json_list = list(os.walk(json_root))[0][2]

        file_list = []
        for js in json_list:
            with open(pathlib.Path(json_root) / js) as f:
                metal_info = json.load(f)

            file_name = metal_info['fileInfo']['nameUpdated']
            if not pathlib.Path.is_file(self.image_root / file_name):
                continue

            file_dict = {
                'id': metal_info['fileInfo']['id'],
                'name': file_name,
                'set': metal_info['kfolds']['split1'].lower()
            }
            file_dict.update(metal_info['labels'])
            file_list.append(file_dict)

        df = pd.DataFrame(file_list).set_index('id')
        if set_name.lower() != 'all':
            df = df[df['set'] == set_name]

        if exclude_category:
            self.dataset = df[~df[target_category].isin(exclude_category)]

        if self.dataset is None or self.dataset.empty:
            raise ValueError('empty dataset!!')
        else:
            print(f'Set {set_name} Raw Images: {len(self.dataset)}')

        self.species = self.dataset[target_category].tolist()
        unique_species = sorted(list(set(self.species)), key=lambda x: '#' if x == 'Ghost' else x)
        self.species_classes_map = {spec: unique_species.index(spec) for spec in unique_species}
        self.classes = list(self.species_classes_map.keys())

        if set_name == 'train' and flip_image:
            self.transform_func = default_train_transform
        else:
            self.transform_func = default_val_transform

    def __len__(self):
        return len(self.dataset)

    def get_image_path(self, idx):
        return self.image_root / self.dataset.iloc[idx]['name']

    def __getitem__(self, idx):
        img_path = self.get_image_path(idx)
        img = io.imread(img_path)
        img = img if self.transform_func is None else self.transform_func(img)

        target = self.species[idx]
        return img, self.species_classes_map[target]

    def show_image(self, idx):
        return torchvision.utils.make_grid(self[idx][0]).numpy().transpose([1, 2, 0])
