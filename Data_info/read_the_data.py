import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import logging



def data_information(dataset):
    df = pd.read_csv(dataset)

    df.head(10)
    # print()
    # logging.info(f" info of dataset is : {df.info()} ")
    # print()
    # logging.info(f" shape of dataset is : {df.shape}")
    # print()
    return df