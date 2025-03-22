# BEST: Bilaterally Expanding Subtrace Tree for Event Sequence Prediction

This is the implementation accompanying the paper submitted to the BPM Conference 2025 in Seville, Spain (Track II: Engineering)

## Framework

We provide a prediction framework to forecast future activities of running business processes. The framework itself builds upon a tree structure of bilaterally expanding subtrace patterns extracted from business process event logs. The method is capable of predicting the next activities and the complete remaining trace of a running process.

## Setup

We implemented our approach as a python module (`best4ppm`) and provided an overall forecasting script to reproduce our experimental results.
To setup the environment for running our code, we provide a `pyproject.toml` file (requires python>3.12) from which the needed dependencies can be gathered with `pip` via (execute from the project directory):

`python -m pip install .`

## Usage

The codebase consists of our module `best4ppm` and different scripts using the module for dataset manipulation (`BPI2012_conversions.py`), event log metric extraction (`log_characteristics.py`) and the experiments for the prediction of next activities and remaining traces (best_prediction.py).

We also provide a set of config files, which are populated with the needed setup to reproduce our experimental results (`general_config.yaml`, `model_config.yaml`, `data_config.yaml`).

### Config

The general config sets the parameters for the main prediction loop. You can specify the datasets you want to analyze (`dataset` either as list of multiple or string of a single dataset), the evaluation strategy (i.e., cross-validation with `cv_folds` > 1 or single split with `cv_folds`==1 alongside the desired 'train_pct' specifying the share of cases you want to use for training) and the model configuration (`model_config`). For multiprocessing you can set the number of cores you want to use for the evaluation (`ncores`).

The general config file specifies the is linked to the remaining config files. With `dataset` you access the different data configurations matched by the dataset name. The data config file specifies the dataset filename (`file_name`) and the relevant column identifiers (`case_identifier`, `activity_identifier`, `timestamp_identifier`).

The model config file holds the model-specific parameters for model training. Our model has different parameters: 

- `max_pattern_size_train`/`max_pattern_size_eval`: specifying the depth of the tree (train) and the maximum traversal depth in the evaluation loop (eval). The depth is specified via the maximum allowed subtrace pattern size where the (pattern size - 1)/2 is the depth of the tree.

- 'process_stage_width_percentage': The main parameter for the number of process stages. The number of process stages is statically determined by calculating the process stage width via the percentage of the maximum trace length we see in the training data. A value of zero results in `n` process stages of width 1 with `n` being the maximum trace length we see in the training data, i.e., n BEST models. A value of 1 results in a single process stage, i.e., one trained BEST model.

- `task`: the tasks you want to perform. `nap` performs Next Activity Prediction and `rtp` performs Remaining Trace Prediction (can also be passed as a list of both tasks)

- `min_freq`: cutoff frequency for subtrace patterns. A value close to zero prevents filtering of subtrace patterns. We set this to 10e-15 in our experiments.

- `break_buffer`: the predicted sequence length at which the prediction loop is terminated in terms of `break_buffer` times the maximum trace length we see in the training dataset. We set this to 1.2 in our experiments.

- `prune_func`: A prune function for tree pruning. This is not applied/implemented currently

- `filter_sequences`: this filters the padded dummy activity tokens from the predicted sequences for evaluation. Should be set to `True` for a sound evaluation of predictive performance.

-   Load a desired event log data set. We provide the analyzed datasets in the `data/` folder (Helpdesk[[D1]](#D1), BPI2012[[D2]](#D2) (with variations: Full, Sub and WC), BPI2020 - Travel Permits[[D3]](#D3) and Road Traffic Fines[[D4]](#D4)).
-   Adjust the predefined model setup and constants if necessary.
-   Generate the Process-aware network structure with the provided set of functions.
-   Preprocess the data with the provided set of functions.
-   Query the Process-aware Bayesian Network with the framework provided by the `bnlearn`[[1]](#1) package we rely on in our implementations. We provide exemplary queries that can be used and adapted by practitioners.

## Scripts

We provide multiple scripts for execution. The `2x_model_training_x.R` files are for model training for each application (Next Activity/Remaining Trace Prediction, Overall Case Duration Class Prediction or the Process Query System). Files (`3x_evaluation_x.R`) are for evaluating the three applications. `10_preliminaries.R` loads the required extensions as well as the `custom_BN_functions.R`.

## References
<a id="1">[1]</a> Scutari,  Marco (2007). bnlearn: Bayesian Network Structure Learning,  Parameter Learning and Inference. CRAN: Contributed Packages (Link: <https://cran.r-project.org/package=bnlearn>).

## Dataset References

<a id="D1">[D1]</a> Verenich, Ilya (2016). Helpdesk. Mendeley (Link: <https://doi.org/10.17632/39BP3VV62T.1>).

<a id="D2">[D2]</a> van Dongen, Boudewijn (2012). BPI Challenge 2012. Eindhoven University of Technology (Link: <https://data.4tu.nl/articles/_/12689204/1>)

<a id="D3">[D3]</a> van Dongen, Boudewijn (2020). BPI Challenge 2020: Travel Permit Data. 4TU.Centre for Research Data (Link: <https://data.4tu.nl/articles/dataset/BPI_Challenge_2020_Travel_Permit_Data/12718178/1>).

<a id="D4">[D4]</a> de Leoni, M. and Mannhardt, Felix (2015). Road Traffic Fine Management Process. Eindhoven University of Technology (Link: <https://data.4tu.nl/articles/_/12683249/1>).
