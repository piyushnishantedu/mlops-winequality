# Load train and test
# train algo
# save metrics param

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from get_data import read_params
import argparse
import jsonlib
import json
import math

def evaluate_metrics(actual, pred):
    corr_matrix = numpy.corrcoef(actual, pred)
    corr = corr_matrix[0,1]
    mae = mean_squared_error(actual, pred)
    r2 = corr**2
    rmse = math.sqrt(mae)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["alpha"]
    target = config["base"]["target_col"]
    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)
    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target,axis=1)
    test_x =  test.drop(target, axis=1)
    lr = ElasticNet(alpha=alpha,
                     l1_ratio=l1_ratio, 
                     random_state=random_state)
    lr.fit(train_x,train_y)
    predict_qualities = lr.predict(test_x)
    (rmse, mae, r2) = evaluate_metrics(test_y, predict_qualities)

    return rmse, mae, r2



if __name__ = "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    parsed_args = args.parse_args()
   train_and_evaluate(config_path=parsed_args.config)
