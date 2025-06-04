#Question 3(20 Points):

import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import pandas as pd

# Array of 3 inputs and all possible combinations
X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

# training
y = np.array([1 if x.sum() % 2 == 0 else 0 for x in X])

# Train Perceptron
percep = lm.Perceptron(max_iter=500, tol=1e-3, random_state=0)
percep.fit(X, y)
percep_preds = percep.predict(X)

# Part 2 Perceptron learning
percep_result = pd.DataFrame(X, columns=["Input1", "Input2", "Input3"])
percep_result["Expected"] = y
percep_result["Predicted"] = percep_preds

# Percep result
print("Perceptron Result")
print(percep_result)

# Part 3 MLP training
mlp = nn.MLPClassifier(hidden_layer_sizes=(4,), max_iter=500, random_state=1)
mlp.fit(X, y)
mlp_preds = mlp.predict(X)

# MLP Result
mlp_results = pd.DataFrame(X, columns=["Input1", "Input2", "Input3"])
mlp_results["Expected"] = y
mlp_results["Predicted"] = mlp_preds

print("Multi-Layer Perceptron Result")
print(mlp_results)