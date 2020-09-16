# Saving Nature Capstone Project

## 1. Conda environment setup

`conda env create --file environment.yaml -n savingnature`

To export your environment: `conda env export --no-builds --from-history > environment.yaml`

## 2. Download raw data from OneDrive

Download the entire `Camera Traps` folder from [OneDrive](https://onedrive.live.com/?authkey=%21AsXX7LZF08enJgU&id=3E2E9FF710ECC0B5%21173040&cid=3E2E9FF710ECC0B5). 
Rename and move folder to `./data/raw/brazil_amld`.

## 3. Run python script to clean directory

`python3 src/data/clean_directory.py brazil_amld`
