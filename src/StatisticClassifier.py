import pandas as pd
import numpy as np

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


df = pd.read_csv("DSL-StrongPasswordData.csv")
for i in np.arange(80.0,100.0,0.1):
    t = StatisticClassifier(df, i)
    t.evaluate()
