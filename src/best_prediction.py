from best4ppm.models.best import BESTPredictor
from best4ppm.data.sequencedata import SequenceData
from best4ppm.util.config_utils import read_config
from best4ppm.eval.evaluator import NAPEvaluator
from best4ppm.eval.evaluator import RTPEvaluator
from best4ppm.util.model_logging import log_to_csv
from best4ppm.util.parallelization import warmup_worker_pool
import best4ppm.util.pruning as pruning
import time
from itertools import product
import os
import numpy as np
import random
import pandas as pd

from best4ppm.util.logging import init_logging
logger = init_logging(__name__, 'main.log')

import psutil
import threading
from contextlib import contextmanager

TRACE_MEMORY = True

def main():

    if TRACE_MEMORY:
        memory_tracking = {}
        _process = psutil.Process(os.getpid())
    else:
        memory_tracking = None
        _process = None

    general_config = read_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best4ppm', 'configs', 'general_config.yml'))
    data_configs = read_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best4ppm', 'configs', 'data_configs.yml'))
    model_configs = read_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best4ppm', 'configs', 'model_configs.yml'))
    export_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'export')
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    os.makedirs(export_path, exist_ok=True)
    
    if general_config['parallelization_lib']=='joblib':
        warmup_worker_pool(ncores=general_config['ncores'])

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
        additional_params['parallelization_lib'] = general_config['parallelization_lib']

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
                
                with track_peak_ram(name="train", process=_process, results=memory_tracking, trace_memory=TRACE_MEMORY):
                    
                    folds = data.train_test_split(train_pct=general_config.get('train_pct'), cv=general_config.get('cv_folds'))
                    times['data_prep_time'] = time.perf_counter()
                    base_cv_hash = cv_hashes[comb_idx]

                    model_params = dict(zip(list(model_config.keys()), [param for param in combination]))
                    if model_config is None:
                        raise KeyError('desired model config not found in model_config.yml')
                    
                    fold_models = list()
                    
                    all_fold_times = {fold_idx: dict() for fold_idx in range(len(folds))}

                    for fold_idx, fold in enumerate(folds):

                        data_train, data_test = fold
                        fold_times = all_fold_times[fold_idx]
                        fold_times['run_start_time'] = time.perf_counter()

                        fold_models.append(perform_run_train(data_train, data_test, model_params, general_config, fold_times))
                        
                    for time_key in all_fold_times[0].keys():
                        times[time_key] = [all_fold_times[fold_idx][time_key] for fold_idx in range(len(folds))]

                for eval_pattern_size_idx, eps in enumerate(model_params['max_pattern_size_eval']):
                    
                    if model_params['weights'] == ['None']:
                        s_method_weight_combs = [[m, weight] for m, weights in zip(model_params['selection_method'], [None]) for weight in sorted([None])]
                    else:
                        s_method_weight_combs = [[m, weight] for m, weights in zip(model_params['selection_method'], model_params['weights']) for weight in sorted(weights)]
                    
                    for s_method_weight_comb_idx, s_method_weight_comb in enumerate(s_method_weight_combs):

                        s_method = s_method_weight_comb[0]
                        weight = s_method_weight_comb[1]
                    
                        for fold_idx, best in enumerate(fold_models):
                            
                            with track_peak_ram(name="test", process=_process, results=memory_tracking, trace_memory=TRACE_MEMORY):

                                run_log_params_metrics = {}
                                
                                run_log_params_metrics['process_stage_width'] = best._abs_process_stage_width
                                run_log_params_metrics['n_process_stages'] = len(best._stages)
                                
                                model_params_eval = {key: model_params[key] for key in model_params.keys() if key not in ['max_pattern_size_eval', 'selection_method', 'weight']}
                                model_params_eval['max_pattern_size_eval'] = eps
                                model_params_eval['selection_method'] = s_method
                                model_params_eval['weight'] = weight
                                
                                run_log_params_metrics['random_seed'] = additional_params['seed']
                                run_log_params_metrics['parallelization_lib'] = additional_params['parallelization_lib']


                                model_and_general_params = {**model_params,
                                                            'max_pattern_size_eval': eps, 
                                                            'selection_method': s_method, 
                                                            'weight': weight, 
                                                            **{'model_config':general_config['model_config'],
                                                            'ncores':general_config['ncores'],
                                                            'cv_folds':general_config.get('cv_folds'),
                                                            'train_pct':general_config.get('train_pct')},
                                                            'dataset':additional_params['dataset']}

                                for key, value in model_and_general_params.items():
                                    run_log_params_metrics[key] = value

                                run_log_params_metrics['base_cv_hash'] = base_cv_hash
                                run_log_params_metrics['cv_hash'] = f'{base_cv_hash}_eps{eval_pattern_size_idx}_s{s_method_weight_comb_idx}'

                                perform_run_test(best, model_params_eval, general_config, times, run_log_params_metrics, fold_idx=fold_idx)
                                
                            if TRACE_MEMORY:
                                run_log_params_metrics['peak_ram_train'] = memory_tracking['train']['peak_ram_gb']
                                run_log_params_metrics['peak_ram_test'] = memory_tracking['test']['peak_ram_gb']

                            log_to_csv(csv_file=os.path.join(export_path, 'model_params_metrics.csv'), params_metrics=run_log_params_metrics)
                        
            else:
                
                with track_peak_ram(name="train", process=_process, results=memory_tracking, trace_memory=TRACE_MEMORY):
                    data_train, data_test = data.train_test_split(train_pct=general_config.get('train_pct'), cv=general_config.get('cv_folds'))
                    times['data_prep_time'] = time.perf_counter()
                    
                    model_params = dict(zip(list(model_config.keys()), [param for param in combination]))

                    if model_config is None:
                        raise KeyError('desired model config not found in model_config.yml')
                    
                    times['run_start_time'] = time.perf_counter()

                    best = perform_run_train(data_train, data_test, model_params, general_config, times)

                for eps in model_params['max_pattern_size_eval']:
                    
                    if model_params['weights'] == ['None']:
                        s_method_weight_combs = [[m, weight] for m, weights in zip(model_params['selection_method'], [None]) for weight in sorted([None])]
                    else:
                        s_method_weight_combs = [[m, weight] for m, weights in zip(model_params['selection_method'], model_params['weights']) for weight in sorted(weights)]
                    
                    for s_method_weight_comb_idx, s_method_weight_comb in enumerate(s_method_weight_combs):

                        s_method = s_method_weight_comb[0]
                        weight = s_method_weight_comb[1]
                    
                        with track_peak_ram(name="test", process=_process, results=memory_tracking, trace_memory=TRACE_MEMORY):

                            run_log_params_metrics = {}

                            run_log_params_metrics['process_stage_width'] = best._abs_process_stage_width
                            run_log_params_metrics['n_process_stages'] = len(best._stages)

                            model_params_eval = {key: model_params[key] for key in model_params.keys() if key not in ['max_pattern_size_eval', 'selection_method', 'weight']}
                            model_params_eval['max_pattern_size_eval'] = eps
                            model_params_eval['selection_method'] = s_method
                            model_params_eval['weight'] = weight
                                    

                            run_log_params_metrics['random_seed'] = additional_params['seed']
                            run_log_params_metrics['parallelization_lib'] = additional_params['parallelization_lib']

                            model_and_general_params = {**model_params,
                                                        'max_pattern_size_eval': eps, 
                                                        'selection_method': s_method, 
                                                        'weight': weight, 
                                                        **{'model_config':general_config['model_config'],
                                                        'ncores':general_config['ncores'],
                                                        'cv_folds':general_config.get('cv_folds'),
                                                        'train_pct':general_config.get('train_pct')},
                                                        'dataset':additional_params['dataset']}

                            for key, value in model_and_general_params.items():
                                run_log_params_metrics[key] = value
                            
                            run_log_params_metrics['base_cv_hash'] = 'single_fold_run'
                            run_log_params_metrics['cv_hash'] = 'single_fold_run'

                            perform_run_test(best, model_params_eval, general_config, times, run_log_params_metrics)
                        
                        if TRACE_MEMORY:
                                run_log_params_metrics['peak_ram_train'] = memory_tracking['train']['peak_ram_gb']
                                run_log_params_metrics['peak_ram_test'] = memory_tracking['test']['peak_ram_gb']
                        
                        log_to_csv(csv_file=os.path.join(export_path, 'model_params_metrics.csv'), params_metrics=run_log_params_metrics)

def perform_run_train(data_train, data_test, model_params_train, general_config, times):

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
                        prune_func=prune_func,
                        choice_tracker_keys_nap=general_config['choice_tracker_keys_nap'],
                        choice_tracker_keys_rtp=general_config['choice_tracker_keys_rtp'],
                        parallelization_lib=general_config['parallelization_lib'])
    
    best.load_data(data_train, data_test)
    
    best.prepare_train()
    best.fit()
    
    best.prepare_test(act_encoder=data_train.act_encoder, filter_sequences=model_params_train['filter_sequences'])

    times['fitting_time'] = time.perf_counter()

    return best

def perform_run_test(model: BESTPredictor, model_params_eval, general_config, times, param_metric_dict, fold_idx = None):
    
    times['prediction_start_time_nap'] = time.perf_counter()
    times['prediction_start_time_rtp'] = time.perf_counter()

    if 'nap' in model_params_eval['task']:
        nap_predictions, nap_pred_dur, nap_pred_convert_dur = model.predict(task='nap', 
                                                                            selection_method=model_params_eval['selection_method'],
                                                                            weight=model_params_eval['weight'], 
                                                                            eval_pattern_size=model_params_eval['max_pattern_size_eval'],
                                                                            break_buffer=model_params_eval['break_buffer'], 
                                                                            filter_tokens=model_params_eval['filter_sequences'], 
                                                                            ncores=general_config['ncores'])
        times['nap_pred_duration'] = nap_pred_dur
        times['nap_pred_convert_duration'] = nap_pred_convert_dur
        times['nap_finish_time'] = time.perf_counter()
        
        nap_eval = NAPEvaluator(pred=nap_predictions, actual=model.data_test.next_activities)
        none_share = nap_eval.get_nan_share()
        nap_acc = nap_eval.calc_accuracy_score()
        nap_balanced_acc = nap_eval.calc_balanced_accuracy_score()
        logger.info(f'None share of predictions: {none_share:.4f}')
        logger.info(f'NAP accuracy: {nap_acc:.4f}')
        logger.info(f'NAP balanced accuracy: {nap_balanced_acc:.4f}')

        param_metric_dict['none_share'] = none_share
        param_metric_dict['nap_accuracy'] = nap_acc
        param_metric_dict['nap_balanced_accuracy'] = nap_balanced_acc
        
        nap_n_events_per_second = len(model.data_test.relevant_prefixes)/nap_pred_dur
        param_metric_dict['nap_n_events_per_second'] = nap_n_events_per_second
        
        times['nap_eval_time'] = time.perf_counter()
        times['prediction_start_time_rtp'] = time.perf_counter()
    
    if 'rtp' in model_params_eval['task']:
        rtp_predictions, rtp_pred_dur, rtp_pred_convert_dur = model.predict(task='rtp', 
                                                                            selection_method=model_params_eval['selection_method'],
                                                                            weight=model_params_eval['weight'], 
                                                                            eval_pattern_size=model_params_eval['max_pattern_size_eval'], 
                                                                            break_buffer=model_params_eval['break_buffer'], 
                                                                            filter_tokens=model_params_eval['filter_sequences'], 
                                                                            ncores=general_config['ncores'])
        times['rtp_pred_duration'] = rtp_pred_dur
        times['rtp_pred_convert_duration'] = rtp_pred_convert_dur
        times['rtp_finish_time'] = time.perf_counter()
        
        rtp_eval = RTPEvaluator(pred=rtp_predictions, actual=model.data_test.full_future_sequences)
        ndls = rtp_eval.calc_ndls(ncores=general_config['ncores'])
        logger.info(f'RTP similarity: {ndls:.4f}')
        param_metric_dict['rtp_similarity'] = ndls
        
        rtp_n_traces_per_second = len(model.data_test.relevant_prefixes)/rtp_pred_dur
        param_metric_dict['rtp_n_traces_per_second'] = rtp_n_traces_per_second

        horizons = general_config.get('eval_horizons')
        if horizons:
            for horizon in horizons:
                horizon_similarity = rtp_eval.calc_ndls(horizon=horizon, ncores=general_config['ncores'])
                param_metric_dict[f'rtp_similarity_h_{horizon}'] = horizon_similarity

        times['rtp_eval_time'] = time.perf_counter()

    times['run_end_time'] = time.perf_counter()
    
    calc_times = calc_runtimes(recorded_times=times, fold_idx=fold_idx)
    
    for key, value in calc_times.items():
        param_metric_dict[key] = value

def calc_runtimes(recorded_times: dict, fold_idx: int = None):
    
    calculated_runtimes = dict()
    calculated_runtimes['prep_duration'] = recorded_times['data_prep_time'] - recorded_times['start_time']
    
    if fold_idx is not None: # we have multiple fold runs
        calculated_runtimes['total_fit_duration'] = recorded_times['fitting_time'][-1] - recorded_times['run_start_time'][0]
        calculated_runtimes['fit_duration_per_fold'] = recorded_times['fitting_time'][fold_idx] - recorded_times['run_start_time'][fold_idx]
        
    else: # we have a single fold run    
        calculated_runtimes['total_fit_duration'] = recorded_times['fitting_time'] - recorded_times['run_start_time']
        calculated_runtimes['fit_duration_per_fold'] = calculated_runtimes['total_fit_duration']
    
    calculated_runtimes['nap_duration'] = recorded_times['nap_finish_time'] - recorded_times['prediction_start_time_nap']
    calculated_runtimes['nap_eval_duration'] = recorded_times['nap_eval_time'] - recorded_times['nap_finish_time']
    calculated_runtimes['rtp_duration'] = recorded_times['rtp_finish_time'] - recorded_times['prediction_start_time_rtp']
    calculated_runtimes['rtp_eval_duration'] = recorded_times['rtp_eval_time'] - recorded_times['rtp_finish_time']
    
    if fold_idx is not None:
        calculated_runtimes['total_run_time'] = recorded_times['run_end_time'] - recorded_times['run_start_time'][0]
    else:
        calculated_runtimes['total_run_time'] = recorded_times['run_end_time'] - recorded_times['run_start_time']
    
    return calculated_runtimes

@contextmanager
def track_peak_ram(name: str, process, results: dict, poll_interval=0.05, trace_memory=False):
    if trace_memory:
        peak = 0
        stop = threading.Event()

        def poll():
            nonlocal peak
            while not stop.is_set():
                peak = max(peak, process.memory_info().rss)
                time.sleep(poll_interval)

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()

        yield

        stop.set()
        poller.join()
        results[name] = {
            "peak_ram_bytes": peak,
            "peak_ram_gb": round(peak / 1024 / 1024 / 1024, 4),
        }
    else:
        yield

if __name__=='__main__':
    main()
    