import pandas as pd

def load_data():
    reference_data = pd.read_csv("data/reference.csv")
    current_data = pd.read_csv("data/current.csv")

    return reference_data, current_data