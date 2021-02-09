import os
import pathlib

import torchvision
from skimage import io
from torch.utils.data import Dataset

from constant import default_train_transform, default_val_transform, get_image_category


class ClassificationDataset(Dataset):
    def __init__(self, set_name, root_dir, exclude_category=('Exclude',), target_category='species_binary',
                 flip_image=False):
        """
        The class is used to feed image data into pytorch model, inherited from torch Dataset.
        Args:
            set_name: str
                train, test, or val
            root_dir: str
                image root folder
            exclude_category: tuple
                categories to be excluded
            target_category: str
                possible classified categories:
                    species_binary (binary classification),
                    class,
                    order,
                    family,
                    genus,
                    species_new
            flip_image: bool
                flag for train set only, if True rotate the image
        """

        self.root_dir = pathlib.Path(root_dir) / set_name
        species_category = get_image_category(target_category=target_category)

        raw_files = [file.split('.')[0].split('_') for file in os.listdir(self.root_dir) if file.endswith('.jpg')]
        self.image_files = [[species_category.get(image[-8], 'Exclude')] + image for image in raw_files]
        if exclude_category:
            self.image_files = [image for image in self.image_files if image[0] not in exclude_category]

        self.species = [spec[0] for spec in self.image_files]
        # Make sure 'Ghost' will be the first if it exist
        unique_species = sorted(list(set(self.species)), key=lambda x: '#' if x == 'Ghost' else x)
        self.species_classes_map = {spec: unique_species.index(spec) for spec in unique_species}
        self.classes = list(self.species_classes_map.keys())

        self.dataset = self.image_files

        if set_name == 'train' and flip_image:
            self.transform_func = default_train_transform
        else:
            self.transform_func = default_val_transform

        self.data_set_species = [spec[0] for spec in self.dataset]
        self.cam_trap = [spec[4] if len(spec) == 13 else '_'.join(spec[4:6]) for spec in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def get_image_path(self, idx):
        return self.root_dir / '{}.jpg'.format('_'.join(self.dataset[idx][1:]))

    def __getitem__(self, idx):
        img_path = self.get_image_path(idx)
        img = io.imread(img_path)
        img = img if self.transform_func is None else self.transform_func(img)

        target = self.dataset[idx][0]
        return img, self.species_classes_map[target]

    def show_image(self, idx):
        return torchvision.utils.make_grid(self[idx][0]).numpy().transpose([1, 2, 0])
