"""
Utility Module
--------------
Utility functions for logging model evaluation metrics and saving predictions.
This module provides helper functions to facilitate the evaluation and logging of 
machine learning model performance. It includes functionality for computing standard 
and normalized metrics, saving model predictions, and appending results to a JSON 
log file for tracking experiments.

Functions:
    - write_log_results: Logs the results of a model run, including metrics and 
      training history, to a JSON file.
    - log_results: Computes evaluation metrics, saves predictions, and logs the 
      results for a given model.

Usage:
    Import this module and use the `log_results` function to evaluate a model's 
    predictions, save the results, and log them for future reference.
"""

import json

import datetime
import pandas as pd
import numpy as np
from IPython.display import display
from recs.metrics import get_metrics, get_norm_metrics
from pathlib import Path

results_json = Path('results.json')

def write_log_results(modelname, dataset_size, history=None, hyperparams=None, **metrics):
    """
    Logs the results of a model run to a JSON file named 'results.json'.

    This function creates a log entry containing the model name, the current timestamp,
    and any additional metrics provided via keyword arguments. If a Keras history object is
    supplied, it also records the last loss values from the training history. The new log
    entry is appended to the existing list of results in 'results.json'. If the file does
    not exist or contains invalid JSON, a new list is created.
    
    Args:
        modelname (str): The name or identifier of the current model or execution.
        history (keras.callbacks.History, optional): The Keras History object containing training metrics.
        **metrics: Arbitrary keyword arguments representing additional custom metrics.

    Returns:
        None:
        Writes a log to `results.json` in the `project_folder`.
    """
    
    try:
        with results_json.open() as fin:
            results = json.load(fin)
    except (FileNotFoundError, json.JSONDecodeError):
        results = []

    final_results = {
        'modelname': modelname,
        'dataset_size': dataset_size,
        'timestamp': datetime.datetime.now().strftime('%F %H:%M')}
    
    if metrics:
        # Add all metrics
        final_results.update(metrics)
    
    if history:
        # Add the final value for every key
        final_results.update({k: round(history.history[k][-1], 4) 
                              for k in history.history.keys()})
        
    if hyperparams:
        final_results.update(hyperparams)
    
    results.append(final_results)

    with results_json.open('w') as fout:
        json.dump(results, fout, indent=4)

    display(pd.Series(final_results).to_frame(modelname))

def log_results(y_true, 
                y_pred, 
                imprs, 
                modelname:str, 
                history=None,
                hyperparams=None,
                metrics: list = [
                    'auc', 'mean_mrr', 'ndcg@5;10'
                    ],
                norm_metrics: list=[
                    "mean_epc", 
                    'mean_intra_list_diversity', 
                    'mean_surprisal',
                    'prediction_coverage'
                    ], 
                dataset_size=None,
                **norm_kwargs):
    """
    Computes evaluation metrics and writes log results for a given model.

    This function calculates standard evaluation metrics and normalized metrics by invoking the 
    get_metrics and get_norm_metrics functions, respectively. The results from both metric calculations 
    are then combined and passed as keyword arguments to the write_log_results function along with the 
    model name.
    Parameters:
        y_true (array-like): The actual or target values.
        y_pred (array-like): The predicted values from the model.
        imprs (any): Additional metric-related information used in normalization,
                     typically representing importance scores or similar values.
        modelname (str): The name or identifier of the model whose results are being logged.
    Returns:
        None
        Displays a pd.DataFrame with the logged metrics.
        Logs the computed metrics by calling write_log_results.
        Saves the predictions using the name.
    """    
    save_path = f"../.data/{modelname.replace(' ', '_')}.npy"
    np.save(save_path, y_pred)
    print("Saved model predictions here:", save_path)

    write_log_results(
        modelname,
        dataset_size,
        history,
        hyperparams,
        **get_metrics(y_true, y_pred, metrics),
        **get_norm_metrics(y_pred, y_true, imprs, 
                           norm_metrics=norm_metrics, 
                           dataset_size=dataset_size, 
                           kwargs=norm_kwargs))