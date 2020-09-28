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

