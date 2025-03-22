from best4ppm.data.sequencedata import SequenceData
from best4ppm.util.config_utils import read_config
import os
import logging
import yaml

logging.basicConfig(
        # filename='characteristics_log.log',
        encoding='utf-8', 
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        )

def main():

    general_config = read_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best4ppm', 'configs', 'general_config.yml'))
    data_configs = read_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best4ppm', 'configs', 'data_configs.yml'))
    
    for dataset in general_config['dataset']:
        try:
            data_config = data_configs[dataset]
        except KeyError as e:
            e.args = (f'desired datset {dataset} not found in data_config.yml', )
            raise

        data = SequenceData.from_csv(load_path=data_config['load_path'],
                                    case_identifier=data_config['case_identifier'],
                                    activity_identifier=data_config['activity_identifier'],
                                    timestamp_identifier=data_config['timestamp_identifier'])
                
        log_characteristics = data.get_characteristics()

        export_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'export', 'characteristics')

        os.makedirs(export_path, exist_ok=True)

        with open(os.path.join(export_path, f'{dataset}_characteristics.csv'), 'w') as f:
            yaml.dump(log_characteristics, f, sort_keys=False)

if __name__=='__main__':
    main()
    