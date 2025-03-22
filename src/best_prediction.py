from best4ppm.models.best import BESTPredictor
from best4ppm.data.sequencedata import SequenceData
from best4ppm.util.config_utils import read_config
from best4ppm.eval.evaluator import NAPEvaluator
from best4ppm.eval.evaluator import RTPEvaluator
from best4ppm.util.model_logging import log_to_csv
import best4ppm.util.pruning as pruning
import time
from itertools import product
import os
import logging
import numpy as np
import random
import pandas as pd

logging.basicConfig(
        # filename='predict_log.log',
        encoding='utf-8', 
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        )

def main():

    general_config = read_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best4ppm', 'configs', 'general_config.yml'))
    data_configs = read_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best4ppm', 'configs', 'data_configs.yml'))
    model_configs = read_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best4ppm', 'configs', 'model_configs.yml'))
    export_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'export')
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    os.makedirs(export_path, exist_ok=True)

    for dataset in general_config['dataset']:

        try:
            data_config = data_configs[dataset]
        except KeyError as e:
            e.args = (f'desired datset {dataset} not found in data_config.yml', )
            raise
        
        model_config = model_configs[general_config['model_config']]
        model_config_train = {key: model_config[key] for key in model_config.keys() if key!='max_pattern_size_eval'}
        model_config_eval = {key: model_config[key] for key in model_config.keys() if key!='max_pattern_size_train'}
        config_combinations = list(product(*model_config.values()))
        additional_params = dict()
        
        cv_hashes = [random.getrandbits(128) for _ in range(0, len(config_combinations))]

        additional_params['seed'] = general_config['seed']
        additional_params['dataset'] = dataset

        max_pattern_size_eval = max(model_config_eval['max_pattern_size_eval'][0])

        if max_pattern_size_eval > model_config_train['max_pattern_size_train'][0]:
            raise ValueError('max_pattern_size_train must be higher than maximum max_pattern_size_eval!')

        for comb_idx, combination in enumerate(config_combinations):

            random.seed(additional_params['seed'])
            np.random.seed(additional_params['seed'])
 
            times = dict()
            times['start_time'] = time.perf_counter()
            
            data = SequenceData.from_csv(load_path=os.path.join(data_path, data_config['file_name']),
                                        case_identifier=data_config['case_identifier'],
                                        activity_identifier=data_config['activity_identifier'],
                                        timestamp_identifier=data_config['timestamp_identifier'])
            
            if general_config['cv_folds'] > 1:
                folds = data.train_test_split(train_pct=general_config.get('train_pct'), cv=general_config.get('cv_folds'))
                times['data_prep_time'] = time.perf_counter()
                base_cv_hash = cv_hashes[comb_idx]

                model_params = dict(zip(list(model_config.keys()), [param for param in combination]))
                if model_config is None:
                    raise KeyError('desired model config not found in model_config.yml')
                
                fold_models = list()

                for fold in folds:

                    data_train, data_test = fold
                    times['run_start_time'] = time.perf_counter()

                    fold_models.append(perform_run_train(data_train, data_test, model_params, times))

                for eval_pattern_size_idx, eps in enumerate(model_params['max_pattern_size_eval']):
                        
                    for best in fold_models:

                        run_log_params_metrics = {}
                        
                        run_log_params_metrics['process_stage_width'] = best._abs_process_stage_width
                        run_log_params_metrics['n_process_stages'] = len(best._stages)
                        
                        model_params_eval = {key: model_params[key] for key in model_params.keys() if key!='max_pattern_size_eval'}
                        model_params_eval['max_pattern_size_eval'] = eps
                        run_log_params_metrics['random_seed'] = additional_params['seed']

                        model_and_general_params = {**model_params,
                                                    'max_pattern_size_eval': eps, 
                                                    **{'model_config':general_config['model_config'],
                                                       'ncores':general_config['ncores'],
                                                       'cv_folds':general_config.get('cv_folds'),
                                                       'train_pct':general_config.get('train_pct')},
                                                       'dataset':additional_params['dataset']}

                        for key, value in model_and_general_params.items():
                            run_log_params_metrics[key] = value

                        run_log_params_metrics['base_cv_hash'] = base_cv_hash
                        run_log_params_metrics['cv_hash'] = f'{base_cv_hash}_{eval_pattern_size_idx}'

                        perform_run_test(best, model_params_eval, general_config, times, run_log_params_metrics)
                        log_to_csv(csv_file=os.path.join(export_path, 'model_params_metrics.csv'), params_metrics=run_log_params_metrics)

            else:
                data_train, data_test = data.train_test_split(train_pct=general_config.get('train_pct'), cv=general_config.get('cv_folds'))
                times['data_prep_time'] = time.perf_counter()

                model_params = dict(zip(list(model_config.keys()), [param for param in combination]))

                if model_config is None:
                    raise KeyError('desired model config not found in model_config.yml')

                best = perform_run_train(data_train, data_test, model_params, general_config, additional_params, times)

                for eps in model_params['max_pattern_size_eval']:

                    run_log_params_metrics = {}

                    run_log_params_metrics['process_stage_width'] = best._abs_process_stage_width
                    run_log_params_metrics['n_process_stages'] = len(best._stages)

                    model_params_eval = {key: model_params[key] for key in model_params.keys() if key!='max_pattern_size_eval'}
                    model_params_eval['max_pattern_size_eval':eps]

                    run_log_params_metrics['random_seed'] = additional_params['seed']

                    model_and_general_params = {**model_params,
                                                'max_pattern_size_eval': eps, 
                                                **{'model_config':general_config['model_config'],
                                                   'ncores':general_config['ncores'],
                                                   'cv_folds':general_config.get('cv_folds'),
                                                   'train_pct':general_config.get('train_pct')},
                                                   'dataset':additional_params['dataset']}

                    for key, value in model_and_general_params.items():
                        run_log_params_metrics[key] = value

                    run_log_params_metrics['base_cv_hash'] = base_cv_hash
                    run_log_params_metrics['cv_hash'] = f'{base_cv_hash}_{eval_pattern_size_idx}'

                    perform_run_test(best, model_params_eval, general_config, times, run_log_params_metrics)
                    log_to_csv(csv_file=os.path.join(export_path, 'model_params_metrics.csv'), params_metrics=run_log_params_metrics)

def perform_run_train(data_train, data_test, model_params_train, times):

    if model_params_train['prune_func'] == 'None':
        prune_func = None
    else:
        try:
            prune_func = pruning.load_func('best4ppm.util.pruning.' + model_params_train['prune_func'])
        except AttributeError as e:
            e.add_note(f"Desired pruning function '{model_params_train['prune_func']}' not implemented")
            raise

    best = BESTPredictor(max_pattern_size=model_params_train['max_pattern_size_train'],
                        process_stage_width_percentage=model_params_train['process_stage_width_percentage'],
                        min_freq=model_params_train['min_freq'],
                        prune_func=prune_func)
    
    best.load_data(data_train, data_test)
    
    best.prepare_train()
    best.fit()
    
    best.prepare_test(act_encoder=data_train.act_encoder, filter_sequences=model_params_train['filter_sequences'])

    times['fitting_time'] = time.perf_counter()

    return best

def perform_run_test(model: BESTPredictor, model_params_eval, general_config, times, param_metric_dict):
    
    times['prediction_start_time_nap'] = time.perf_counter()
    times['prediction_start_time_rtp'] = time.perf_counter()

    if 'nap' in model_params_eval['task']:
        nap_predictions = model.predict(task='nap', eval_pattern_size=model_params_eval['max_pattern_size_eval'],
                                        break_buffer=model_params_eval['break_buffer'], 
                                        filter_tokens=model_params_eval['filter_sequences'], 
                                        ncores=general_config['ncores'])
        times['nap_finish_time'] = time.perf_counter()
        
        nap_eval = NAPEvaluator(pred=nap_predictions, actual=model.data_test.next_activities)
        none_share = nap_eval.get_nan_share()
        nap_acc = nap_eval.calc_accuracy_score()
        nap_balanced_acc = nap_eval.calc_balanced_accuracy_score()
        logging.info(f'None share of predictions: {none_share:.4f}')
        logging.info(f'NAP accuracy: {nap_acc:.4f}')
        logging.info(f'NAP balanced accuracy: {nap_balanced_acc:.4f}')

        param_metric_dict['none_share'] = none_share
        param_metric_dict['nap_accuracy'] = nap_acc
        param_metric_dict['nap_balanced_accuracy'] = nap_balanced_acc
        times['nap_eval_time'] = time.perf_counter()
        times['prediction_start_time_rtp'] = time.perf_counter()
    
    if 'rtp' in model_params_eval['task']:
        rtp_predictions = model.predict(task='rtp', 
                                        eval_pattern_size=model_params_eval['max_pattern_size_eval'], 
                                        break_buffer=model_params_eval['break_buffer'], 
                                        filter_tokens=model_params_eval['filter_sequences'], 
                                        ncores=general_config['ncores'])
        times['rtp_finish_time'] = time.perf_counter()
        
        rtp_eval = RTPEvaluator(pred=rtp_predictions, actual=model.data_test.full_future_sequences)
        ndls = rtp_eval.calc_ndls(ncores=general_config['ncores'])
        logging.info(f'RTP similarity: {ndls:.4f}')
        param_metric_dict['rtp_similarity'] = ndls

        horizons = general_config.get('eval_horizons')
        if horizons:
            for horizon in horizons:
                horizon_similarity = rtp_eval.calc_ndls(horizon=horizon, ncores=general_config['ncores'])
                param_metric_dict[f'rtp_similarity_h_{horizon}'] = horizon_similarity

        times['rtp_eval_time'] = time.perf_counter()

    times['run_end_time'] = time.perf_counter()

    calc_times = dict()
    calc_times['prep_duration'] = times['data_prep_time'] - times['start_time']
    calc_times['total_fit_duration'] = times['fitting_time'] - times['run_start_time']
    calc_times['fit_duration_per_fold'] = calc_times['total_fit_duration'] / general_config['cv_folds']
    calc_times['nap_duration'] = times['nap_finish_time'] - times['prediction_start_time_nap']
    calc_times['nap_eval_duration'] = times['nap_eval_time'] - times['nap_finish_time']
    calc_times['rtp_duration'] = times['rtp_finish_time'] - times['prediction_start_time_rtp']
    calc_times['rtp_eval_duration'] = times['rtp_eval_time'] - times['rtp_finish_time']
    calc_times['total_run_time'] = times['run_end_time'] - times['run_start_time']
    for key, value in calc_times.items():
        param_metric_dict[key] = value

if __name__=='__main__':
    main()
    