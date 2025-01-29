from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class Regressions:
    def __init__(self, dataset, correlation):
        self.dataset = dataset
        self.correlation = correlation
        self.strength = 0
        self.is_linear = self.linear_checker()
        self.reg_model = 'linear' if self.is_linear else 'randomForest'
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataset[:-1], self.dataset[-1], test_size=0.2, random_state=0
        )
        self.y_pred, self.mse, self.r2, self.accuracy, self.importances = self.regressor()

    def linear_checker(self):
        for i in range(self.correlation.shape[0]):
            for j in range(self.correlation.shape[1]):
                if self.correlation[i, j] > 0.8:
                    self.strength += 1

        if self.strength / self.correlation.size > 0.5:
            return True
        else:
            return False

    def regressor(self):
        model = LinearRegression() if self.is_linear else RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        accuracy = model.score(self.X_test, self.y_test)
        importances = model.feature_importances_
        return y_pred, mse, r2, accuracy, importances


