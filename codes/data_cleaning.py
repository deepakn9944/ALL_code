import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import pandas as pd

# Step 1: Load NSL-KDD dataset
file_path = '/content/drive/My Drive/nsl-kdd/KDDTrain+_20Percent.csv'
file_path_test = '/content/drive/My Drive/nsl-kdd/KDDTest+.csv'

nsl_kdd = pd.read_csv(file_path)
nsl_kdd_test = pd.read_csv(file_path_test)

nsl_kdd.head(5)

nsl_kdd_test.info()

nsl_kdd.info()

print('Label distribution Training set:')
print(nsl_kdd['label'].value_counts())
print()
print('Label distribution Test set:')
print(nsl_kdd_test['label'].value_counts())

print(nsl_kdd.isnull().values.any())
print(nsl_kdd_test.isnull().values.any())

missing_columns= [col for col in nsl_kdd.columns if nsl_kdd[col].isnull().sum() > 0]
print(f"Number of missing columns: {missing_columns} ")
print(f"Number of duplicate rows: {nsl_kdd.duplicated().sum()}")
# Removing duplicate rows
nsl_kdd.drop_duplicates(inplace=True)

# Check the shape of the dataset after removing duplicates
print(f"New shape of the dataset: {nsl_kdd.shape}")

missing_columns= [col for col in nsl_kdd_test.columns if nsl_kdd_test[col].isnull().sum() > 0]
print(f"Number of missing columns: {missing_columns} ")
print(f"Number of duplicate rows: {nsl_kdd_test.duplicated().sum()}")
# Removing duplicate rows
nsl_kdd_test.drop_duplicates(inplace=True)

# Check the shape of the dataset after removing duplicates
print(f"New shape of the dataset: {nsl_kdd_test.shape}")

import seaborn as sns
sns.countplot(x=nsl_kdd["label"])

import pandas as pd

# Assuming df is your DataFrame
def check_constant_columns(nsl_kdd_test):
    constant_columns = [col for col in nsl_kdd_test.columns if nsl_kdd_test[col].nunique() == 1]
    return constant_columns

# Example usage
constant_cols = check_constant_columns(nsl_kdd_test)
if constant_cols:
    print(f"Columns with the same value across all rows: {constant_cols}")
else:
    print("No columns have the same value across all rows.")

nsl_kdd.describe(include="object")

nsl_kdd_test.describe(include="object")

#Apply LabelEncoding to all the categorical columns (those with data type 'object')

from sklearn.preprocessing import LabelEncoder

def LabelEncoding(df):
    for col in df.columns:
        if df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])

LabelEncoding(nsl_kdd)
LabelEncoding(nsl_kdd_test)
nsl_kdd.head()

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# Split the dataset into features and labels
X_initial = nsl_kdd.drop('label', axis=1)
y_initial = nsl_kdd['label']

X_main = nsl_kdd_test.drop('label', axis=1)
y_main = nsl_kdd_test['label']

# Normalize data
scaler = MinMaxScaler()
X_initial_scaled = scaler.fit_transform(X_initial)
X_main_scaled = scaler.transform(X_main)
