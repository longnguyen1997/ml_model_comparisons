# Machine Learning — Model Comparisons #

This project analyzes accuracy differences between four provided classification models given in the ```scitkit-learn``` library; we examine an SVM with the default radial basis kernel function, the multivariate Bernoulli model, the passive-aggressive model, and the multilayered perceptron neural network model.

## Data ##

Two datasets were used in 10-fold cross-validation, with permission of UCI's ML Repository. The first involved congressional voting records from 1984, in which samples (people) were classified as either Democrats or Republicans. The second involved choosing mathematics heuristics to quickly prove a theorem, with this second dataset being much more complex and large in size relative to the first.

Datasets can be found below.

*Congressional Voting Records Data Set*: 
```https://archive.ics.uci.edu/ml/datasets/congressional+voting+records```

*First-order theorem proving Data Set*: 
```https://archive.ics.uci.edu/ml/datasets/First-order+theorem+proving```

## Results ##

The multilayered perceptron was the most accurate of the four models used in analyzing these datasets, with 99% maximum accuracy when dealing with voting records. Given the complexity of the mathematics dataset, this model only attained 53% accuracy, though still higher than any of the others.

Modification of default parameters left untouched may yield more accurate results. For a visual comparison of how each model analyzed the mathematics dataset, please refer to ```model_comparison.png```.
