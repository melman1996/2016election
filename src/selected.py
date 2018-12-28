from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ann_visualizer.visualize import ann_viz
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
                results[result[3]][0] = result[7]
            elif result[5] == "Hillary Clinton":
                results[result[3]][1] = result[7]
            elif result[5] == "Ben Carson":
                results[result[3]][2] = result[7]
            elif result[5] == "Donald Trump":
                results[result[3]][3] = result[7]
            elif result[5] == "John Kasich":
                results[result[3]][4] = result[7]
            elif result[5] == "Marco Rubio":
                results[result[3]][5] = result[7]
            elif result[5] == "Ted Cruz":
                results[result[3]][6] = result[7]

    for key in results:
        if results[key][0] > results[key][1]:
            results[key][0] = 1
            results[key][1] = 0
        elif results[key][0] < results[key][1]:
            results[key][0] = 1
            results[key][1] = 0

        best = max(results[key][2:])
        if best > 0:
            for i in range(2, len(results[key])):
                if results[key][i] == best:
                    results[key][i] = 1
                else:
                    results[key][i] = 0
    return np.array([results[key] for key in results])


if __name__ == "__main__":
    np.random.seed(2137)

    # read dataset from csv
    county_facts = pd.read_csv("county_facts.csv")
    primary_resaults = pd.read_csv("primary_results.csv")
    poland = pd.read_csv("poland.csv")

    X = county_facts.iloc[:, [5, 7, 8, 9, 10, 12, 17, 18, 19, 20, 21, 22, 23, 25, 27, 28, 31, 34, 40, 44]].values
    Y = normalize_results(primary_resaults)
    X_predict = poland.iloc[:, [5, 7, 8, 9, 10, 12, 17, 18, 19, 20, 21, 22, 23, 25, 27, 28, 31, 34, 40, 44]].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # scale values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # create NN
    model = Sequential()
    model.add(Dense(output_dim=19, init='uniform', activation='relu', input_dim=20))
    model.add(Dense(output_dim=11, init='uniform', activation='relu'))
    model.add(Dense(output_dim=7, init='uniform', activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # fit the model
    model.fit(X_train, Y_train, batch_size=10, epochs=100)
    # evaluate NN
    scores = model.evaluate(X_test, Y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    prediction = model.predict(X_predict, batch_size=None, verbose=0, steps=None)
    print(prediction)
