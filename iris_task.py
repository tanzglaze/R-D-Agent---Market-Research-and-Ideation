"""
Iris Task
A simple built-in task using the classic Iris dataset.
Perfect for testing the full R&D loop locally — no downloads needed.
"""

TASK_DESCRIPTION = """
Classify Iris flowers into 3 species (setosa, versicolor, virginica).
This is a 4-feature, 150-sample, 3-class classification problem.
Maximize cross-validated accuracy (cv=5).
"""

DATA_LOADING_CODE = """
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target
"""
