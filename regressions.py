import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def linear_checker(dataset):
    correlation = np.absolute(np.corrcoef(dataset))
    strength = 0
    for i in range(correlation.shape[0]):
        for j in range(correlation.shape[1]):
            if correlation[i, j] > 0.8:
                strength += 1

    if strength/correlation.size >= 0.5:
        return True
    else:
        return False


