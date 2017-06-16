import numpy as np
from ml_functions import model_comparison_classification as model_compare
from ml_functions import plot_model_comparison as model_plot
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def get_lines_labels(filename):

    def get_labels(lines):
        labels = []
        for l in lines:
            if l[-1] == 1:
                labels.append(0)
            elif l[-2] == 1:
                labels.append(1)
            elif l[-3] == 1:
                labels.append(2)
            elif l[-4] == 1:
                labels.append(3)
            elif l[-5] == 1:
                labels.append(4)
            else:
                labels.append(5);

        return labels

    with open(filename) as f:
        lines = f.readlines()
    lines = [line.strip().split(",") for line in lines]
    lines = [[float(n) for n in l] for l in lines]
    labels = get_labels(lines)
    lines = [l[:-6] for l in lines]
    return (lines, labels)

lines_labels_train = get_lines_labels('data/ml-prove/train.csv')
lines_labels_test = get_lines_labels('data/ml-prove/test.csv')

theorem_data_train = ([np.array(l) for l in lines_labels_train[0]],
                      lines_labels_train[1])
theorem_data_test = ([np.array(l) for l in lines_labels_test[0]],
                lines_labels_test[1])

# Compare the different classifiers.
theorem_models =  model_compare(10, theorem_data_train)

model_plot(theorem_models)

# Take the average of the k bins.
theorem_models = {k: np.average(theorem_models[k])
                  for k in theorem_models}

k = theorem_models.keys()
v = theorem_models.values()

# On previous runs, best model reported
# as multilayered perceptron.
best_model = (k[v.index(max(v))], max(v))

clf = MLPClassifier()
clf.fit(theorem_data_train[0], theorem_data_train[1])
results = clf.predict(theorem_data_test[0])
print accuracy_score(theorem_data_test[1], results)
