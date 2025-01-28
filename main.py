import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

matplotlib.use('TkAgg')

path = "/Users/mohammad/.cache/kagglehub/datasets/mathurinache/sleep-dataset/versions/1"
dataset = pd.read_csv(path + "/sleep-dataset.csv")

# cleaning dataset
dataset.replace('?', np.nan, inplace=True)
dataset.dropna(inplace=True)

cov_matrix = dataset.cov()  # covariance
correlation_matrix = dataset.corr(method='pearson')  # correlation

# plotting
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

