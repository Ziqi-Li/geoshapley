
![PyPI](https://img.shields.io/pypi/v/fastgwr)
![GitHub](https://img.shields.io/github/license/Ziqi-Li/fastgwr)


# GeoShapley
A game theory approach to measure spatial effects from machine learning models. GeoShapley is built on Shapley value and Kernel SHAP estimator.

### Installation:

GeoShapley can be installed from PyPI:

```bash
$ pip install geoshapley
```

```bash
$ from geoshapley import GeoShapleyExplainer
$
$ Specify a background sample
$ background_X = X_coords.sample(100).values
$ 
$ explainer = GeoShapleyExplainer(model.predict, background_X)
$ rslt = explainer.explain(X_geo)
$ rslt.summary_plot()
```

