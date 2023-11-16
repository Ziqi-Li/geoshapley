
![PyPI](https://img.shields.io/pypi/v/geoshapley)
![GitHub](https://img.shields.io/github/license/Ziqi-Li/geoshapley)


# GeoShapley
A game theory approach to measure spatial effects from machine learning models. GeoShapley is built on Shapley value and Kernel SHAP estimator.

### Installation:

GeoShapley can be installed from PyPI:

```bash
$ pip install geoshapley
```

```python
from geoshapley import GeoShapleyExplainer

#Specify a background sample
background_X = X_geo.sample(100).values

explainer = GeoShapleyExplainer(model.predict, background_X)

#Explain the data
rslt = explainer.explain(X_geo)

#Make a shap-style summary plot
rslt.summary_plot()
```

