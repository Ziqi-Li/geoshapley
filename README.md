
![PyPI](https://img.shields.io/pypi/v/geoshapley)
![GitHub](https://img.shields.io/github/license/Ziqi-Li/geoshapley)


# GeoShapley

<img src="https://github.com/Ziqi-Li/geoshapley/assets/5518908/b450b5b3-fd59-41d8-a64c-fb202f492302" width="500">



A game theory approach to measuring spatial effects from machine learning models. GeoShapley is built on Shapley value and Kernel SHAP estimator.

### Recent Updates
- 04/2025 - Some speed up (2x - 10x) with vectorization and sparse matrix

### Installation:

GeoShapley can be installed from PyPI:

```bash
$ pip install geoshapley
```

To install the latest version from Github:

```bash
$ pip install git+https://github.com/ziqi-li/geoshapley.git
```

### Example:

GeoShapley can explain any model that takes tabular data + spatial features (e.g., coordinates) as the input. Examples of natively supported models include:
1. XGBoost/CatBoost/LightGBM/Random Forest
2. Microsoft's FLAML AutoML (see example in notebook folder)
3. MLP or other `scikit-learn` modules.
4. Tabular Deep Learning models such as [TabNet](https://github.com/dreamquark-ai/tabnet)
6. [Explainable Boosting Machine](https://github.com/interpretml/interpret)
7. Statistical models: OLS/Gaussian Process/GWR

Other models can be supported by defining a helper function model.predict() to wrap around their original models' prediction or inference functions.

Currently, spatial features (e.g., coordinates, or other encodings) need to be put as the last columns of your `pandas.DataFrame`(`X_geo`). 


Below shows an example on how to explain a trained MLP model. More examples can be found at the notebooks folder.

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

#Make partial dependence plots of the primary (non-spatial) effects
mlp_rslt.partial_dependence_plots()

#Calculate spatially varying explanations
mlp_svc = mlp_rslt.get_svc()
```


### References:
- Li, Z. (2024). GeoShapley: A Game Theory Approach to Measuring Spatial Effects in Machine Learning Models. Annals of the American Association of Geographers. Open access at: [https://www.tandfonline.com/doi/full/10.1080/24694452.2024.2350982](https://www.tandfonline.com/doi/full/10.1080/24694452.2024.2350982)
- Li, Z. (2022). Extracting spatial effects from machine learning model using local interpretation method: An example of SHAP and XGBoost. Computers, Environment and Urban Systems, 96, 101845. Open access at: [https://www.sciencedirect.com/science/article/pii/S0198971522000898](https://www.sciencedirect.com/science/article/pii/S0198971522000898)


### A list of recent papers that applied GeoShapley:
- Peng, Z., Ji, H., Yuan, R., Wang, Y., Easa, S. M., Wang, C., ... & Zhao, X. (2025). Modeling and spatial analysis of heavy-duty truck CO2 using travel activities. Journal of Transport Geography, 124, 104158.
- Ke, E., Zhao, J., & Zhao, Y. (2025). Investigating the influence of nonlinear spatial heterogeneity in urban flooding factors using geographic explainable artificial intelligence. Journal of Hydrology, 648, 132398.
- Foroutan, E., Hu, T., & Li, Z. (2025). Revealing key factors of heat-related illnesses using geospatial explainable AI model: A case study in Texas, USA. Sustainable Cities and Society, 122, 106243.
- Wu, R., Yu, G., & Cao, Y. (2025). The impact of industrial structural transformation in the Yangtze River economic belt on the trade-offs and synergies between urbanization and carbon balance. Ecological Indicators, 171, 113165.

