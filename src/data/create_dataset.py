# import libraries
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys


def train_val_test_split(df_ds, split_ratio = [0.7, 0.1, 0.2], random_state=0):
    # train, val, test split
    df_train_val, df_test = train_test_split(df_ds, test_size=split_ratio[2],
                                             stratify=df_ds.species_binary, random_state=random_state)
    df_train, df_val = train_test_split(df_train_val, test_size=round(split_ratio[1]/(split_ratio[0]+split_ratio[1]), 3),
                                        stratify=df_train_val.species_binary, random_state=random_state)
    # append splits back together
    df_train = df_train.assign(split="train")
    df_val = df_val.assign(split="val")
    df_test = df_test.assign(split="test")
    df_ds_split = df_train.append(df_val).append(df_test)
    print(f"dataset split: train {split_ratio[0]}%, val {split_ratio[1]}%, test {split_ratio[2]}%")
    return df_ds_split


def extract_df_ds(input_file):
    # load location csv
    df = pd.read_csv(input_file, dtype="str")
    # extract df for images only
    df_image = df[(df.data_type == "image") & (df.species_binary != "Exclude")].reset_index(drop=True)
    df_ds = df_image.iloc[:, np.r_[:10, 12:20]]
    # generate ds_id
    df_ds.index.rename("ds_id", inplace=True)
    df_ds = df_ds.reset_index()
    df_ds.ds_id = df_ds.ds_id.astype("str").str.zfill(4)
    df_ds['label'] = df_ds.species_binary.replace({"Ghost": 0, "Animal": 1})
    return df_ds


def extract_to_csv(df_ds_split, path_dataset, output_file):
    # file_path_ds
    df_ds_split['file_path_ds'] = path_dataset + "/" + \
                                  df_ds_split.apply(lambda x: '%s/%s' % (x['split'], x['label']), axis=1)

    # export df to csv
    df_ds_split.to_csv(output_file, index=False)
    print(f"{output_file} saved!")
    return df_ds_split


def copy_to_ds_dir(ds_dict):
    # copy all data from raw data directory to interim data directory with renamed file name
    for k, v in ds_dict.items():
        os.makedirs(v['file_path_ds'], exist_ok=True)
        shutil.copyfile(v['file_path_new'] + '/' + v['file_name_new'], v['file_path_ds'] + "/" + v['file_name_new'])
    print("dataset creation complete!")


def create_dataset(model, dataset_name, location_name, split_ratio=[0.7, 0.1, 0.2], random_state=0):
    # set directories
    #os.chdir("../")  # change current working directory to root directory of project
    path_interim = f"./data/interim/{location_name}"
    path_dataset = f"./data/processed/{model}/{dataset_name}"
    os.makedirs(path_interim, exist_ok=True)
    os.makedirs(path_dataset, exist_ok=True)
    input_file = f"{path_interim}/{location_name}.csv"
    output_file = f"{path_dataset}/{dataset_name}.csv"

    df_ds = extract_df_ds(input_file)
    df_ds_split = train_val_test_split(df_ds, split_ratio, random_state)
    df_ds_split = extract_to_csv(df_ds_split, path_dataset, output_file)
    ds_dict = df_ds_split.to_dict('index')
    print(f"copying from {path_interim} to {path_dataset}\nreorganizing directories...")
    copy_to_ds_dir(ds_dict)


def main():
    model = sys.argv[1]
    dataset_name = sys.argv[2]
    location_name = sys.argv[3]
    split_ratio = [float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])]
    random_state = int(sys.argv[7])
    create_dataset(model, dataset_name, location_name, split_ratio, random_state)


if __name__ == "__main__":
    # execute only if run as a script
    main()