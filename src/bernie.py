from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sys


def normalize_results(primary_results):
    results = {}
    for county in county_facts.iloc[:, 0].values:
        results[county] = [0] * 7

    for result in primary_resaults.iloc[:, :].values:
        if result[3] in results:
            if result[5] == "Bernie Sanders":
                results[result[3]][0] = result[6]
            elif result[5] == "Hillary Clinton":
                results[result[3]][1] = result[6]
            elif result[5] == "Ben Carson":
                results[result[3]][2] = result[6]
            elif result[5] == "Donald Trump":
                results[result[3]][3] = result[6]
            elif result[5] == "John Kasich":
                results[result[3]][4] = result[6]
            elif result[5] == "Marco Rubio":
                results[result[3]][5] = result[6]
            elif result[5] == "Ted Cruz":
                results[result[3]][6] = result[6]
    to_return = []
    for key in results:
        best = max(results[key])
        if best == results[key][0]:
            to_return.append(1)
        else:
            to_return.append(0)
    return np.array(to_return)


if __name__ == "__main__":
    np.random.seed(2137)

    # read dataset from csv
    county_facts = pd.read_csv("county_facts.csv")
    primary_resaults = pd.read_csv("primary_results.csv")
    X = county_facts.iloc[:, 3:].values
    Y = normalize_results(primary_resaults)

    model = LogisticRegression()
    rfe = RFE(model, 3)
    fit = rfe.fit(X, Y)
    print("Feature Ranking: {}".format(fit.ranking_))
    for rank in fit.ranking_:
        print(rank)

    sys.exit()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # scale values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # create NN
    model = Sequential()
    model.add(Dense(output_dim=30, init='uniform', activation='relu', input_dim=51))
    model.add(Dense(output_dim=15, init='uniform', activation='relu'))
    model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # fit the model
    model.fit(X_train, Y_train, batch_size=10, epochs=100)
    # evaluate NN
    scores = model.evaluate(X_test, Y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))