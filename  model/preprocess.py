import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize

def load_and_preprocess(csv_path='../data/Comparison Stats.csv'):

    df = pd.read_csv(csv_path)

