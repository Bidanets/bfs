from sklearn import datasets 
       
class Data_Preprocessor:
    def __init(self):
        return 1
        
    def Prepare_data(self, dataset):    
        if dataset == 'diabetes':
            data = datasets.load_diabetes()
            names_of_features = ['AGE', 'SEX', 'BMI',
                                 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
            
        if dataset == 'boston':
            data = datasets.load_boston()
            names_of_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                                 'AGE', 'DIS', 'RAD', 'TEX', 'PTRATIO', 'B', 
                                 'LSTAT']
            
        X = data.data
        
        y = data.target
        
        size_of_train_data = int(len(X) * 0.8)
        
        X_train = X[:size_of_train_data]
        y_train = y[:size_of_train_data]
        
        X_test = X[size_of_train_data:]
        y_test = y[size_of_train_data:]
        
        return X_train, y_train, X_test, y_test, names_of_features
    