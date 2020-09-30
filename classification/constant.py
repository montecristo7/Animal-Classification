import pandas as pd
from torchvision import transforms

# regulated_size = (1440, 1920)
regulated_size = 300

default_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=regulated_size),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

default_val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=regulated_size),
    transforms.ToTensor(),
])


def get_image_category(base_category='species_new', target_category='species_binary'):
    image_category = pd.read_csv('brazil_amld.csv')
    image_category = image_category[image_category.file_type == 'jpg']
    cateory = image_category[['species', 'species_new', 'species_binary', 'genus', 'family', 'order', 'class']]
    cateory = cateory.drop_duplicates()

    return cateory.set_index(base_category).to_dict()[target_category]
