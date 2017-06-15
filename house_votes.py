import numpy as np
from ml_functions import model_comparison_classification as model_compare

# features = 'n': 0, 'y': 1, '?': 2}
# labels = 'republican = 1', 'democrat = 0'

with open('data/house-votes-84.txt') as f:
    lines = f.readlines()
lines = [line.strip().split(",") for line in lines]

labels = [l[0] for l in lines]
lines = [l[1:] for l in lines]

for l in lines:
    for i in range(len(l)):
        if l[i] == 'n': l[i] = 0
        elif l[i] == 'y': l[i] = 1
        else: l[i] = 2

data_x = [np.array(l[1:]) for l in lines]
voting_data = (data_x, labels)

# Compare the different classifiers.
voting_models =  model_compare(10, voting_data)
# Take the average of the k bins.
voting_models = {k: np.average(voting_models[k])
                 for k in voting_models}

k = voting_models.keys()
v = voting_models.values()

best_model = (k[v.index(max(v))], max(v))
print best_model
