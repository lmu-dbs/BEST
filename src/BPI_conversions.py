import pandas as pd
import os
from best4ppm.util.config_utils import read_config
from best4ppm.util.logging import init_logging

from util.paths import CONFIG_PATH, DATA_PATH

def convert_BPI_2012(bpi2012_raw_path, conversions_export_path):

    bpi2012_data = pd.read_csv(bpi2012_raw_path)

    bpi2012_full = bpi2012_data.copy()
    bpi2012_full['concept:name'] = bpi2012_full['concept:name'] + '_' + bpi2012_full['lifecycle:transition']

    bpi2012_complete = bpi2012_data[bpi2012_data['lifecycle:transition'] == 'COMPLETE']
    bpi2012_w = bpi2012_full[bpi2012_full['concept:name'].apply(lambda x: x.split('_')[0])=='W']
    bpi2012_w_complete = bpi2012_w[bpi2012_w['lifecycle:transition'] == 'COMPLETE']

    bpi2012_full.to_csv(os.path.join(conversions_export_path, 'BPI2012_Full.csv'), index=False)
    bpi2012_complete.to_csv(os.path.join(conversions_export_path, 'BPI2012_C.csv'), index=False)
    bpi2012_w.to_csv(os.path.join(conversions_export_path, 'BPI2012_W.csv'), index=False)
    bpi2012_w_complete.to_csv(os.path.join(conversions_export_path, 'BPI2012_WC.csv'), index=False)

def convert_BPI_2017(bpi2017_raw_path, conversions_export_path):

    bpi2017_data = pd.read_csv(bpi2017_raw_path)

    bpi2017_full = bpi2017_data.copy()
    bpi2017_full['concept:name'] = bpi2017_full['concept:name'] + '_' + bpi2017_full['lifecycle:transition']

    bpi2017_full.to_csv(os.path.join(conversions_export_path, 'BPI2017_Full.csv'), index=False)
    

def main():

    logger = init_logging(__name__)

    data_configs = read_config(
        os.path.join(
            CONFIG_PATH,
            "data_configs.yml",
        )
    )

    logger.info(f"Creating BPI2012 conversions in {os.path.abspath(DATA_PATH)}")

    convert_BPI_2012(
        os.path.join(DATA_PATH, data_configs["BPI2012"]["file_name"]), DATA_PATH
    )

    logger.info("BPI2012 conversions created")
    
    logger.info(f"Creating BPI2017 conversions in {os.path.abspath(DATA_PATH)}")

    convert_BPI_2017(
        os.path.join(DATA_PATH, data_configs["BPI2017"]["file_name"]), DATA_PATH
    )

    logger.info("BPI2017 conversions created")


if __name__ == "__main__":
    main()