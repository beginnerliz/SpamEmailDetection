
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import re
from collections import Counter

from datetime import datetime

import numpy as np
import yaml
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import *
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

test_size = 12585
feature_dim = 10000

raw_data = pd.read_csv('RAW.csv')
gpt_data = pd.read_csv("GPT.csv")

raw_data["body"] = raw_data["body"].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
gpt_data["body"] = gpt_data["body"].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))


def createWordIndices(train_set: pd.DataFrame) -> List[str]:
    word_storage = dict()
    for i, (row_id, row) in enumerate(tqdm(train_set.iterrows(), total=len(train_set))):
        content = row["body"]
        words = content.split()
        for word in words:
            if word not in word_storage:
                word_storage[word] = 0
            word_storage[word] += 1

    # select top 10000 words with the highest frequency
    word_freq = [(k, v) for k, v in word_storage.items()]
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
    word_freq = word_freq[:feature_dim]

    word_indices = [x[0] for x in word_freq]

    # Save word index
    with open("word_indices.pkl", "wb") as f:
        pkl.dump(word_indices, f)


class SpamDetectorTraditional():
    def __init__(self, word_indices_path: str) -> None:
        with open(word_indices_path, "rb") as f:
            word_indices = pkl.load(f)
        self.feature_dim = len(word_indices)

        self.word_to_index = {word: i for i, word in enumerate(word_indices)}
        self.index_to_word = {i: word for i, word in enumerate(word_indices)}

        self.features = None
        self.labels = None
        self.test_features = None
        self.test_labels = None

    def toFeature(self, email: str) -> np.array:
        feature = np.zeros(self.feature_dim)
        content = str(email).lower()
        content = re.sub(r'\W', ' ', content)

        words = content.split()

        word_counter = Counter(words)
        for word, count in word_counter.items():
            if word in self.word_to_index:
                index = self.word_to_index[word]
                feature[index] = count

        return feature
    
    def setTrainSet(self, train_set: pd.DataFrame) -> None:
        self.features, self.labels = self.__processDataset(train_set)

    def setTestSet(self, test_set: pd.DataFrame) -> None:
        self.test_features, self.test_labels = self.__processDataset(test_set)

    def __processDataset(self, email_dataframe: pd.DataFrame) -> np.ndarray:
        features = np.zeros((len(email_dataframe), self.feature_dim))
        labels = np.array(email_dataframe["label"])

        for i in tqdm(range(len(email_dataframe)), "Making features"):
            email = email_dataframe.iloc[i]["body"]
            features[i] = self.toFeature(email)

        return features, labels

    def train(self, model_class: Any, model_save_path: str, **model_params) -> None:
        print("Training...")
        model = model_class(**model_params)
        model.fit(self.features, self.labels)
        with open(model_save_path, "wb") as f:
            pkl.dump(model, f)

        return model

    def eval(self, model: Any, gpt_portion: float, model_name: str) -> np.ndarray:
        print("Evaluating...")

        # Output accuracy, precision, recall, F1
        pred_labels = model.predict(self.test_features)
        accuracy = accuracy_score(self.test_labels, pred_labels)
        precision = precision_score(self.test_labels, pred_labels)
        recall = recall_score(self.test_labels, pred_labels)
        f1 = f1_score(self.test_labels, pred_labels)
        confusion = confusion_matrix(self.test_labels, pred_labels)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")

        with open(f"Results.csv", "a") as out_file:
            out_file.write("{},{},{},{},{},{},{}\n".format(
                model_name,
                datetime.now().strftime('%Y%m%d-%H%M%S'),
                gpt_portion,
                accuracy,
                precision,
                recall,
                f1,
            ))

        # Plot confusion matrix with numbers and axis labels
        plt.imshow(confusion, cmap='magma', interpolation='None')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0, 1], ["ham", "spam"])
        plt.yticks([0, 1], ["ham", "spam"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, confusion[i, j], ha='center', va='center', color='red')
        plt.savefig(f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
        plt.show()



if __name__ == "__main__":
    test_set = raw_data[-test_size:]

    for gpt_portion in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        train_set = pd.concat([raw_data[:-test_size], gpt_data[:int(gpt_portion * len(gpt_data))]])

        createWordIndices(train_set)

        trainer = SpamDetectorTraditional("word_indices.pkl")
        trainer.setTrainSet(train_set)
        trainer.setTestSet(test_set)

        # Train and eval with NB
        model = trainer.train(MultinomialNB, "NaiveBayes.pkl")
        trainer.eval(model, gpt_portion, "NB")

        # Train and eval with RandomForest
        params = {"n_estimators": 100, "max_depth": 4, "criterion": "gini"}
        model = trainer.train(RandomForestClassifier, "RandomForest.pkl")
        trainer.eval(model, gpt_portion, "RF")

        # Train and eval with DT
        params = {"criterion": "gini", "max_depth": 10}
        model = trainer.train(DecisionTreeClassifier, "DecisionTree.pkl", **params)
        trainer.eval(model, gpt_portion, "DecisionTree")


    
    

