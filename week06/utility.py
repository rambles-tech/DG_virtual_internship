import logging
import os
import subprocess
import yaml
import pandas as pd
import datetime 
import gc
import re

def read_config_file(filepath):
    with open(filepath, 'r') as stream:
        try:
            return yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            logging.error(exc)

def col_header_val(df,table_config):
    df = df.drop(df.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace('[^\w]','_',regex=True)
    df.columns = list(map(lambda x: x.strip('_'), list(df.columns)))
    expected_col = list(map(lambda x: x.lower(),  table_config['columns']))
    received_col = list(map(lambda x: x.lower(), list(df.columns)))
    #df = df.reindex(sorted(df.columns), axis=1)
    
    if len(received_col) == len(expected_col) and set(expected_col)  == set(received_col):
        logging.info("column name and column length validation passed")
        return 1
    else:
        logging.info("column name and column length validation failed")
        mismatched_columns_file = list(set(df.columns).difference(expected_col))
        logging.info("Following File columns are not in the YAML file",mismatched_columns_file)
        missing_YAML_file = list(set(expected_col).difference(df.columns))
        logging.info("Following YAML columns are not in the file uploaded",missing_YAML_file)
        logging.info(f'df columns: {df.columns}')
        logging.info(f'expected columns: {expected_col}')
        return 0
