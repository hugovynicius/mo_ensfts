from sklearn.preprocessing import StandardScaler, MinMaxScaler

class ScaleData():
    def __init__(self,x):
        self.x = x

        # debug attributes
        self.name = 'Scales data using standardization or minmax'
        self.shortname = 'standardization-minmax'

    def __str__(self):
        return self.name

    def scale_data(self):
        '''
    	scales data using standardization or minmax

        Parameters
        ==========
        X_train: pandas df. Feature data already encoded in Real space and clean
        at this stage.

        Returns
        ==========
        the scaled data and the object for future queries or test set.
        '''

        scaler_std = StandardScaler()
        scaler_std = scaler_std.fit(self.x)
        x_std = scaler_std.transform(self.x)
        scaler_min_max = MinMaxScaler(feature_range=(0, 1))
        scaler_min_max = scaler_min_max.fit(self.x)
        x_min_max = scaler_min_max.transform(self.x)

        return x_std, x_min_max, scaler_std, scaler_min_max

