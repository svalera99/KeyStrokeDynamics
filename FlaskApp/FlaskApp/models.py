import numpy as np
import pandas as pd


def prepare_data(data, subject):

    user_data = data.loc[data.subject == subject, "H.period":"H.Return"].values
    imposter_data = data.loc[data.subject != subject, :].groupby("subject").head(8).loc[:, "H.period":"H.Return"].values

    y_train = np.array([1]*len(user_data)+[0]*len(imposter_data))
    x_train = np.array(list(user_data)+list(imposter_data))

    shuffle_index = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

    train_proportion = 0.8
    train_test_cut = int(len(x_train)*train_proportion)

    x_train, x_test, y_train, y_test = \
    x_train[:train_test_cut], \
    x_train[train_test_cut:], \
    y_train[:train_test_cut], \
    y_train[train_test_cut:]

    x_train = x_train.transpose()
    y_train = y_train.reshape(1,y_train.shape[0])
    x_test = x_test.transpose()
    y_test = y_test.reshape(1,y_test.shape[0])

    return (x_train, x_test, y_train, y_test)


def sigmoid(z):
    
    return 1.0 / (1.0 + np.exp(-z))


def initialize(dim):

    w = np.zeros((dim,1))
    b = 0
    
    assert (w.shape == (dim,1))
    assert (isinstance(b, float) or isinstance(b,int))
    
    return w,b


def propagate(w, b, X, Y):
    
    m = X.shape[1]
    
    z = np.dot(w.T,X)+b
    A = sigmoid(z)
    print(A.shape)
    print(z.shape)
    print(Y.shape)
    cost = -1.0/m*np.sum(Y*np.log(A)+(1.0-Y)*np.log(1.0-A))
    
    dw = 1.0/m*np.dot(X, (A-Y).T)
    db = 1.0/m*np.sum(A-Y)
    
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    
    grads = {"dw": dw, 
             "db":db}
    
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = True):

    costs = []
    
    for i in range(num_iterations):
        
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        if i % 100 == 0:
            costs.append(cost)
            
        if print_cost and i % 100 == 0:
            print ("Cost (iteration %i) = %f" %(i, cost))
            
    grads = {"dw": dw, "db": db}
    params = {"w": w, "b": b}
        
    return params, grads, costs


def predict (w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    A = sigmoid (np.dot(w.T, X)+b)
    
    for i in range(A.shape[1]):
        if (A[:,i] > 0.5): 
            Y_prediction[:, i] = 1
        elif (A[:,i] <= 0.5):
            Y_prediction[:, i] = 0
            
    assert (Y_prediction.shape == (1,m))
    
    return Y_prediction


def model (X_train, Y_train, X_test, Y_test, num_iterations = 1000, learning_rate = 0.5, print_cost = True):
    
    w, b = initialize(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict (w, b, X_test)
    Y_prediction_train = predict (w, b, X_train)
    
    train_accuracy = 100.0 - np.mean(np.abs(Y_prediction_train-Y_train)*100.0)
    test_accuracy = 100.0 - np.mean(np.abs(Y_prediction_test-Y_test)*100.0)
    
    d = {"costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    
    print ("Accuarcy Test: ",  test_accuracy)
    print ("Accuracy Train: ", train_accuracy)
    
    return d



class StatisticClassifier:
    def __init__(self, df, t):
        self.int_df = df.iloc[:, 3:]
        self.df = df
        self.feature_amount = self.int_df.shape[1]
        self.rows = self.df.shape[0]
        self.trueclassified = []
        self.threashold = t
        self.FAR = 0
        self.FRR = 0

    def evaluate(self):
        subjects = self.df["subject"].unique()
        rowsPerUser = self.rows / len(subjects)
        for subject in subjects:
            train_df = self.df.loc[self.df["subject"] == subject, :].iloc[:, 3:]
            mean_vector, deviation_vector = self.train(train_df)

            for inx, row in self.int_df.iterrows():
                working_features = StatisticClassifier.test(row, mean_vector, deviation_vector)

                if working_features > self.feature_amount * self.threashold:
                    self.trueclassified.append(inx)

            self.calculateErrors(subject, rowsPerUser)
            self.trueclassified = []
        print("average FRR is {}".format(self.FRR / len(subjects)))
        print("average FAR is {}".format(self.FAR / len(subjects)))

    def singleClassification(self, train_df, test_vector):
        train_df = train_df.iloc[:, 3:]
        mean_vector, deviation_vector = self.train(train_df)

        working_feature = StatisticClassifier.test(test_vector, mean_vector, deviation_vector)
        return True if working_feature > self.feature_amount * self.threashold else False

    def calculateErrors(self, subject, rowsPerUser):
        trueCorrect = 0
        for inx in self.trueclassified:
            if self.df.iloc[inx, :]["subject"] == subject:
                trueCorrect += 1

        wronglyRejected = rowsPerUser - trueCorrect
        wronglyAccepted = len(self.trueclassified) - trueCorrect

        self.FRR += wronglyRejected / rowsPerUser
        self.FAR += wronglyAccepted / self.rows
        print("Done for {}".format(subject))

    def train(self, train_df):
        mean_vector = []
        deviation_vector = []

        for column in train_df:
            mean_vector.append(np.mean(train_df[column]))
            deviation_vector.append(np.std(train_df[column]))

        return mean_vector, deviation_vector

    @staticmethod
    def test(test_vector, mean_vector, deviation_vector):
        zscore_vector = []
        for inx, feature in enumerate(test_vector):
            zscore_vector.append(np.abs(feature - mean_vector[inx]) / deviation_vector[inx])
        return len(zscore_vector) - len(list(filter(lambda x: x > 1.96, zscore_vector)))