import pandas as pd
import numpy as np
from scipy.spatial.distance import cityblock, euclidean


class BinaryClassificator:
    def __init__(self, df, classificator):
        self.df = df
        self.classificator = classificator
        self.FRR_n = 0
        self.FAR_n = 0
        self.FRR = 0
        self.FAR = 0

    def max_error(self):
        threash_holds = list()
        for i in range(self.partial_df.shape[0]):
            train_vector = self.partial_df.iloc[i]
            mean_vector = np.mean(self.partial_df.copy().drop(i))
            threash_holds.append(self.classificator(train_vector, mean_vector))
        return np.mean(threash_holds)

    def train(self):
        self.mean_vector = self.partial_df.mean().values

    def test(self, x_vector, y_vector):
        predictions = list()

        for vect in range(x_vector.shape[0]):
            if self.classificator(x_vector.iloc[vect], self.mean_vector) < self.error:
                predictions.append(1)
            else:
                predictions.append(0)

        if y_vector.iloc[0].values == 1:
            self.FRR_(predictions, y_vector)
        else:
            self.FAR_(predictions, y_vector)

    def evaluate(self):
        subjects = self.df["subject"].unique()
        for subject in subjects:

            self.partial_df = self.df.loc[self.df["subject"] == subject, "H.period":"H.Return"]
            self.partial_df.index = range(400)
            self.train()
            self.error = self.max_error()

            y = pd.DataFrame(np.ones(self.partial_df.shape[0] + 1))
            self.test(self.partial_df, y)

            imposter_data = self.df.loc[self.df["subject"] != subject, :]
            test_imposter = imposter_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]
            y = pd.DataFrame(np.zeros(test_imposter.shape[0] + 1))
            self.test(test_imposter, y)

        self.show()

    def FAR_(self, predictions, y_vector):
        self.FAR_n += y_vector.shape[0]
        for i in range(len(predictions)):
            if predictions[i] != y_vector.iloc[i].values:
                self.FAR += 1

    def FRR_(self, predictions, y_vector):
        self.FRR_n += y_vector.shape[0]
        for i in range(len(predictions)):
            if predictions[i] != y_vector.iloc[i].values:
                self.FRR += 1

    def show(self):
        print("For classificator {}".format(self.classificator.__name__))
        print("FAR is ", self.FAR / self.FAR_n)
        print("FRR is ", self.FRR / self.FRR_n)


df = pd.read_csv("DSL-StrongPasswordData.csv")
for i in [cityblock, euclidean]:
    m = BinaryClassificator(df, i)
    m.evaluate()