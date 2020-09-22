# import libraries
import os
import shutil
import pandas as pd
import numpy as np
import sys


def extract_dir_info(path_raw, path_interim):
    # extract info from raw directories and file name into dictionary
    location_dict = {}
    file_count = 0
    image_count = 0

    # dict keys
    dir_key = ["cam_trap", "species", "num_animal"]
    metadata_key = ["year", "month", "day", "hour", "minute", "second"]
    key_list = ["file_path", "file_name", "file_type", "data_type",
                "image_id", "num_animal_new", "file_path_new", "file_name_new"]

    for root, dirs, files in os.walk(path_raw):
        for name in files:
            if not name.endswith((".DS_Store")):
                file_id = f"{file_count:04d}"
                file_type = name.split(".")[1].lower()

                # initialise dict
                location_dict[file_id] = dict(zip(dir_key, root.split("/")[4:]))
                species = location_dict[file_id]["species"]
                cam_trap = location_dict[file_id]["cam_trap"]
                # num_animal_new
                num_animal_new = location_dict[file_id].get("num_animal", "01")

                # data_type and etc
                if file_type == "jpg":
                    data_type = "image"
                    image_id = f"{image_count:04d}"
                    metadata = name.split(".")[0].split(" ")
                    # file_name_new
                    file_name_new = "_".join((image_id, cam_trap, species, num_animal_new) \
                                             + tuple(metadata)) + "." + file_type
                    image_count += 1
                else:
                    if file_type == "mp4":
                        data_type = "video"
                    else:
                        data_type = "bbox"
                    image_id = ""
                    metadata = [""] * 6
                    file_name_new = name

                # file_path_new
                file_path_new = "/".join((path_interim, data_type, species))

                # update dict
                value_list = [root, name, file_type, data_type, image_id, num_animal_new, file_path_new, file_name_new]
                location_dict[file_id].update(dict(zip(metadata_key, metadata)))
                location_dict[file_id].update(dict(zip(key_list, value_list)))
                file_count += 1
    return location_dict


def copy_to_clean_dir(location_dict):
    # copy all data from raw data directory to interim data directory with renamed file name
    for k, v in location_dict.items():
        os.makedirs(v['file_path_new'], exist_ok=True)
        shutil.copyfile(v['file_path'] + '/' + v['file_name'], v['file_path_new'] + "/" + v['file_name_new'])
    print("copy to clean directory complete!")


def get_key(value, dictionary):
    # helper function to get key from value in dictionary
    for k, v in dictionary.items():
        if value in v:
            return k


def extract_to_csv(location_dict, csv_name, path_interim):
    # tabulate dictionary into dataframe
    df = pd.DataFrame(columns = ["file_id"] + list(location_dict["0000"].keys()))
    for k, v in location_dict.items():
        location_dict[k].update({"file_id":k})
        df = df.append(location_dict[k], ignore_index=True)

    # mapping of species to binary label
    species_na_list = ["Human", "team", "Unknown", "NonIdent"]
    df["species_binary"] = np.where(df.species=="Ghost", "Ghost",
                                    np.where(df.species.isin(species_na_list), "Exclude", "Animal"))

    # dictionary of species to species subcategories
    species_category = {
        'Rodents':('Guerlinguetus', 'CuniculusPaca', 'Rodent'),
        'Opossums':('MarmosopsIncanus', 'MetachirusMyosurus', 'DidelphisAurita', 'CaluromysPhilander'),
        'Ghost':('Ghost',),
        'Felines':('LeopardusWiedii', 'LeopardusPardalis'),
        'Birds':('Bird', 'PenelopeSuperciliaris', 'LeptotilaRufaxilla'),
        'SmallMammals':('CabassousTatouay', 'TamanduaTetradactyla', 'EuphractusSexcinctus',
                     'ProcyonCancrivorus', 'DasypusNovemcinctus', 'NasuaNasua', 'EiraBarbara'),
        'Reptiles':('SalvatorMerianae',),
        'Canines':('CerdocyonThous', 'CanisLupusFamiliaris'),
        'Exclude':('Unknown', 'Human', 'team', 'NonIdent')}

    # mapping of species to species subcategories
    df['species_category'] = list(map(lambda x: get_key(x, species_category), df.species))

    # export df to csv
    df.to_csv(f"{path_interim}/{csv_name}.csv", index=False)
    print(f"{path_interim}/{csv_name}.csv saved!")


def clean_directory(location_name):
    # set directories
    # os.chdir("../")  # change current working directory to root directory of project
    path_raw = f"./data/raw/{location_name}"
    path_interim = f"./data/interim/{location_name}"
    os.makedirs(path_raw, exist_ok=True)
    os.makedirs(path_interim, exist_ok=True)

    location_dict = extract_dir_info(path_raw, path_interim)
    print(f"copying from {path_raw} to {path_interim}\nreorganizing directories...")
    copy_to_clean_dir(location_dict)
    extract_to_csv(location_dict, location_name, path_interim)


def main():
    location_name = sys.argv[1]
    clean_directory(location_name)


if __name__ == "__main__":
    # execute only if run as a script
    main()