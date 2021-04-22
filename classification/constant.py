import pandas as pd
from torchvision import transforms

# regulated_size = (1440, 1920)
regulated_size = 300, 450


class CropTheBottomStrip(object):

    def __init__(self, top=0, left=0, height=regulated_size[0] * 0.86, width=regulated_size[1]):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, pic):
        return transforms.functional.crop(pic, top=0, left=0, height=300 * 0.86, width=450)


default_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=regulated_size),
    # CropTheBottomStrip(),
    transforms.RandomRotation(degrees=5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

default_val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=regulated_size),
    # CropTheBottomStrip(),
    transforms.ToTensor(),
])


def get_image_category(base_category='species_new', target_category='species_binary',
                       source_file='data_order_class.csv'):
    image_category = pd.read_csv(source_file)
    image_category = image_category[image_category.file_type == 'jpg']
    traditional_category = ['species', 'species_new', 'species_binary', 'genus', 'family', 'order', 'class']
    cateory = image_category[list(set(traditional_category + [base_category, target_category]))]
    cateory = cateory.drop_duplicates()
    if base_category == target_category:
        return {i: i for i in cateory[base_category].to_list()}
    else:
        return cateory.set_index(base_category).to_dict()[target_category]
