from pathlib import Path

import numpy as np
import pandas as pd
import torchvision
from skimage import io
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter

from constant import default_train_transform, default_val_transform

def get_raw_df():
    image_root = Path('image')
    map_file = 'species_list_regrouped_above500.xlsx'

    class_map = pd.read_excel(map_file)
    class_map = class_map[['Labels.Species', 'Regrouped.Binary (above 500)', 'Regrouped.Species (above 500)', 'Regrouped.Class (above 500)', 'Regrouped.Order (above 500)']]

    file_info = pd.read_csv('data_interim_downsampled1000.csv')
    file_info = file_info[
        ['imageInfo.id', 'imageInfo.image.nameUpdated', 'kfolds.split1', 'labels.species', 'labels.binary']]

    file_details = pd.merge(file_info, class_map.reset_index(), how='left', left_on='labels.species',
                            right_on='Labels.Species')

    file_details = file_details.rename(columns={
        'imageInfo.id': 'id',
        'imageInfo.image.nameUpdated': 'filename',
        'kfolds.split1': 'set',
        'Regrouped.Species (above 500)': 'species',
        'Regrouped.Binary (above 500)': 'binary',
        'Regrouped.Class (above 500)': 'class',
        'Regrouped.Order (above 500)': 'order'
    })

    file_details = file_details[['id', 'filename', 'set', 'species', 'binary', 'class', 'order']]
    file_details['set'] = np.where(file_details['set'] == 'test', 'val', 'train')

    print(f'Load raw images: {len(file_details)}')
    return file_details


class ClassificationDataset(Dataset):
    def __init__(self, set_name,
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
        self.image_root = Path('image')
        df = get_raw_df()

        # if set_name.lower() != 'all':  # for test purpose delete this line in the future
        #     df = df[df['set'] == set_name]
        #
        # if exclude_category:
        #     df = df[~df[target_category].isin(exclude_category)]

        if exclude_category:
            df = df[~df[target_category].isin(exclude_category)]

        if set_name.lower() != 'all':  # for test purpose delete this line in the future
            train_val = train_test_split(df, test_size=0.05, random_state=10, stratify=df['species'])
            if set_name.lower() == 'train':
                df = train_val[0]
            else:
                df = train_val[1]

        self.dataset = df

        if self.dataset is None or self.dataset.empty:
            raise ValueError('empty dataset!!')
        else:
            print(f'Set {set_name} Raw Images: {len(self.dataset)}')

        self.species = self.dataset[target_category].tolist()
        unique_species = sorted(list(set(self.species)), key=lambda x: '#' if x == 'Ghost' else x)
        self.species_classes_map = {spec: unique_species.index(spec) for spec in unique_species}
        self.classes = list(self.species_classes_map.keys())

        category_dict = sorted(Counter(self.species).items(), key=lambda x: x[0])
        category_df = pd.DataFrame(category_dict, columns=['Category', '#Images'])
        print('Available categories:')
        print(f'{category_df}')

        if set_name == 'train' and flip_image:
            self.transform_func = default_train_transform
        else:
            self.transform_func = default_val_transform

    def __len__(self):
        return len(self.dataset)

    def get_image_path(self, idx):
        record = self.dataset.iloc[idx]
        ima_path = self.image_root / record.filename
        if Path.is_file(ima_path):
            return ima_path
        else:
            raise ValueError(f'id {idx} is not working')

    def get_original_id(self, idx):
        return self.dataset.iloc[idx]['filename'].split('_')[0]

    def __getitem__(self, idx):
        img_path = self.get_image_path(idx)
        img = io.imread(img_path)
        img = img if self.transform_func is None else self.transform_func(img)

        target = self.species[idx]
        return img, self.species_classes_map[target]

    def show_image(self, idx):
        return torchvision.utils.make_grid(self[idx][0]).numpy().transpose([1, 2, 0])
