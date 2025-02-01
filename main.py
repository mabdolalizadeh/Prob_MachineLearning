import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from regressions import Regressions


matplotlib.use('TkAgg')

dataset = pd.read_csv("sleep-dataset.csv")

# cleaning dataset
dataset.replace('?', np.nan, inplace=True)
dataset.dropna(inplace=True)
dataset = dataset.select_dtypes(include=['number'])

cov_matrix = dataset.cov()  # covariance
correlation_matrix = dataset.corr(method='pearson')  # correlation

# plotting
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

regression = Regressions(dataset, correlation_matrix)

# plot prediction
plt.scatter(regression.y_test, regression.y_pred, color='blue', label='Prediction')
plt.plot([min(regression.y_test), max(regression.y_test)],
         [min(regression.y_test), max(regression.y_test)], 'r--', label='Perfect fit')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title(f"Prediction of {regression.reg_model.capitalize()}")
plt.legend()
plt.show()

# feature importance plot
features = regression.x_train.columns
sorted_indices = np.argsort(regression.importance)[::-1]

plt.figure(figsize=(8, 5))
plt.barh(features[sorted_indices], regression.importance[sorted_indices],
         color='skyblue', align='center')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(f"Feature Importance Of {regression.reg_model.capitalize()}")

print(f"[*] Regression model: {regression.reg_model.capitalize()}")
print(f'[*] Correlation: {correlation_matrix}, It\'s {regression.strength}')
print(f"[*] Covariance matrix: {cov_matrix}")
print(f'[*] MSE: {regression.mse}')
print(f'[*] MMSE: {regression.mmse}')
print(f'[*] R2: {regression.r2}')

with open('final details.txt', 'w') as f:
    f.write(f'[*] Regression model: {regression.reg_model}\n')
    f.write(f'[*] Correlation: {correlation_matrix}\n, It\'s {regression.strength}')
    f.write(f'[*] MSE: {regression.mse}\n')
    f.write(f'[*] MMSE: {regression.mmse}\n')
    f.write(f'[*] R2: {regression.r2}\n')
    f.write(f'[*] Covariance matrix: {cov_matrix}\n')


