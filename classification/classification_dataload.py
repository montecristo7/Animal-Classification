import os
import pathlib

from skimage import io
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

default_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(1440, 1920)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

default_val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(1440, 1920)),
    transforms.ToTensor(),
])


class Classification_Dataset(Dataset):
    def __init__(self, set_name, root_dir, val_size=0.2, random_state=1):
        species_category = {
            'Guerlinguetus': 'Rodents',
            'CuniculusPaca': 'Rodents',
            'Rodent': 'Rodents',
            'MarmosopsIncanus': 'Opossums',
            'MetachirusMyosurus': 'Opossums',
            'DidelphisAurita': 'Opossums',
            'CaluromysPhilander': 'Opossums',
            'Ghost': 'Ghost',
            'LeopardusWiedii': 'Felines',
            'LeopardusPardalis': 'Felines',
            'Bird': 'Birds',
            'PenelopeSuperciliaris': 'Birds',
            'LeptotilaRufaxilla': 'Birds',
            'CabassousTatouay': 'SmallMammals',
            'TamanduaTetradactyla': 'SmallMammals',
            'EuphractusSexcinctus': 'SmallMammals',
            'ProcyonCancrivorus': 'SmallMammals',
            'DasypusNovemcinctus': 'SmallMammals',
            'NasuaNasua': 'SmallMammals',
            'EiraBarbara': 'SmallMammals',
            'SalvatorMerianae': 'Reptiles',
            'CerdocyonThous': 'Canines',
            'CanisLupusFamiliaris': 'Canines',
            'Unknown': 'Exclude',
            'Human': 'Exclude',
            'team': 'Exclude',
            'NonIdent': 'Exclude'
        }
        self.root_dir = root_dir

        raw_files = [file.split('.')[0].split('_') for file in os.listdir(root_dir) if file.endswith('.jpg')]
        self.image_files = [[species_category.get(image[0], 'Exclude')] + image for image in raw_files]

        self.species = [spec[0] for spec in self.image_files]
        self.cam_trap = [spec[2] for spec in self.image_files]

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
        else:
            raise ValueError('Unknown set_name: ' + str(set_name))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = pathlib.Path(self.root_dir) / '{}.jpg'.format('_'.join(self.dataset[idx][1:]))
        img = io.imread(img_path)
        img = img if self.transform_func is None else self.transform_func(img)

        target = self.dataset[idx][0]
        return img, target
