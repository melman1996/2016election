from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import sys


if __name__ == "__main__":
    np.random.seed(2137)

    counties = np.genfromtxt("county_facts.csv", delimiter=",")

    primary = {}
    for i in range(len(counties)):
        primary[counties[i, 0]] = [0, 0, 0, 0, 0, 0, 0]
        counties[i, 32] = counties[i, 32] / 62498               # change annual income to % of max
        counties[i, 53] = counties[i, 53] / 69467.5             # change population per square mile to % of max
        for j in range(len(counties[i])):
            if counties[i, j] > 1:
                counties[i, j] = counties[i, j] / 100           # change % to fractions

    X = counties[:, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 27, 32, 34, 53]]

    ids = np.genfromtxt("primary_results.csv", delimiter=",", usecols=[3])
    results = np.genfromtxt("primary_results.csv", delimiter=",", usecols=[7])
    names = np.genfromtxt("primary_results.csv", delimiter=",", dtype='str', usecols=[5])

    for result, name, county in zip(results, names, ids):
        if county in primary:
            if name == "Bernie Sanders":
                primary[county][0] = result
            elif name == "Hillary Clinton":
                primary[county][1] = result
            elif name == "Ben Carson":
                primary[county][2] = result
            elif name == "Donald Trump":
                primary[county][3] = result
            elif name == "John Kasich":
                primary[county][4] = result
            elif name == "Marco Rubio":
                primary[county][5] = result
            elif name == "Ted Cruz":
                primary[county][6] = result

    for key in primary:
        if primary[key][0] > primary[key][1]:
            primary[key][0] = 1
            primary[key][1] = 0
        elif primary[key][0] < primary[key][1]:
            primary[key][0] = 1
            primary[key][1] = 0

        best = max(primary[key][2:])
        if best > 0:
            for i in range(2, len(primary[key])):
                if primary[key][i] == best:
                    primary[key][i] = 1
                else:
                    primary[key][i] = 0

    Y = np.array([
        primary[key] for key in primary
    ])

    training_length = int(0.7 * len(X))

    model = Sequential()
    input = len(X[0])
    model.add(Dense(50, input_dim=input, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(7, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X[:training_length], Y[:training_length], epochs=150, batch_size=20)
    scores = model.evaluate(X[training_length:], Y[training_length:])
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
