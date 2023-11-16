
![PyPI](https://img.shields.io/pypi/v/geoshapley)
![GitHub](https://img.shields.io/github/license/Ziqi-Li/geoshapley)


# GeoShapley
A game theory approach to measure spatial effects from machine learning models. GeoShapley is built on Shapley value and Kernel SHAP estimator.

### Installation:

GeoShapley can be installed from PyPI:

```bash
$ pip install geoshapley
```

### Example:
```python
from geoshapley import GeoShapleyExplainer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_geo, y, random_state=1)

#Fit a NN model based on training data
mlp_model = MLPRegressor().fit(X_train, y_train)

#Specify a small background data
background = X_train.sample(100).values

#Initilize a GeoShapleyExplainer
mlp_explainer = GeoShapleyExplainer(mlp_model.predict, background)

#Explain the data
mlp_rslt = mlp_explainer.explain(X_geo)

#Make a shap-style summary plot
mlp_rslt.summary_plot()
```

