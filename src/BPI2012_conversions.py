import pandas as pd
import os
from best4ppm.util.config_utils import read_config

def convert_BPI(bpi2012_raw_path, conversions_export_path):

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

def main():
    data_configs = read_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best4ppm', 'configs', 'data_configs.yml'))
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    os.makedirs(data_path, exist_ok=True)
    
    convert_BPI(os.path.join(data_path, data_configs['BPI2012']['file_name']), data_path)

if __name__=='__main__':
    main()

