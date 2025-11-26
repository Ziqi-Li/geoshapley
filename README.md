
![PyPI](https://img.shields.io/pypi/v/geoshapley)
![Pepy-Downloads](https://static.pepy.tech/badge/geoshapley)

# GeoShapley

<img src="https://github.com/Ziqi-Li/geoshapley/assets/5518908/b450b5b3-fd59-41d8-a64c-fb202f492302" width="500">



A game theory approach to measuring spatial effects from machine learning models. GeoShapley is built on Shapley value and Kernel SHAP estimator.

### Recent Updates
- 05/2025 v0.1.0 - Several magnitude of speed-up for more than 10 features by implementing paired sampling from Covert and Lee (2021)

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

#Generate a ranked global feature contribution bar plot of the GeoShapley values.
mlp_rslt.contribution_bar_plot()

#Calculate spatially varying explanations
mlp_svc = mlp_rslt.get_svc()
```

### Visuals:

#### Shap-style summary plot
<img src="https://github.com/user-attachments/assets/16fc24dd-7f6f-4892-a728-75a5e96d4820" width=80% height=80%>

#### Partial dependence plots of the primary (non-spatial) effects

<img src="https://github.com/user-attachments/assets/a7e26870-0749-427a-9783-ecf3e47fd09e" width=80% height=80%>

#### Ranked global feature contribution bar plot of the GeoShapley values

<img src="https://github.com/user-attachments/assets/edb3be7f-e143-4992-8be8-c02e8d36c9fc" width=60% height=60%>



### References:
- Li, Z. (2024). GeoShapley: A Game Theory Approach to Measuring Spatial Effects in Machine Learning Models. Annals of the American Association of Geographers. Open access at: [https://www.tandfonline.com/doi/full/10.1080/24694452.2024.2350982](https://www.tandfonline.com/doi/full/10.1080/24694452.2024.2350982)
- Li, Z. (2025). Explainable AI and Spatial Analysis. In GeoAI and Human Geography, edited by Huang, Wang, Kedron and Wilson. Springer. [https://arxiv.org/abs/2505.00591](https://arxiv.org/abs/2505.00591)
- Li, Z. (2022). Extracting spatial effects from machine learning model using local interpretation method: An example of SHAP and XGBoost. Computers, Environment and Urban Systems, 96, 101845. Open access at: [https://www.sciencedirect.com/science/article/pii/S0198971522000898](https://www.sciencedirect.com/science/article/pii/S0198971522000898)


### A list of recent papers that applied GeoShapley:

<details>
  <summary>Click to expand the list</summary>

- Peng, Z., Ji, H., Yuan, R., Wang, Y., Easa, S. M., Wang, C., ... & Zhao, X. (2025). Modeling and spatial analysis of heavy-duty truck CO2 using travel activities. Journal of Transport Geography, 124, 104158.
- Ke, E., Zhao, J., & Zhao, Y. (2025). Investigating the influence of nonlinear spatial heterogeneity in urban flooding factors using geographic explainable artificial intelligence. Journal of Hydrology, 648, 132398.
- Foroutan, E., Hu, T., & Li, Z. (2025). Revealing key factors of heat-related illnesses using geospatial explainable AI model: A case study in Texas, USA. Sustainable Cities and Society, 122, 106243.
- Chen, Y., Jiao, S., Gu, X., & Li, S. (2025). Decoding the Spatiotemporal Effects of Industrial Clusters on Carbon Emissions in a Chinese River Basin. Journal of Cleaner Production, 145851.
- Wu, R., Yu, G., & Cao, Y. (2025). The impact of industrial structural transformation in the Yangtze River economic belt on the trade-offs and synergies between urbanization and carbon balance. Ecological Indicators, 171, 113165.
- Chen, Y., Ye, Y., Liu, X., Yin, C., & Jones, C. A. (2025). Examining the Nonlinear and Spatial Heterogeneity of Housing Prices in Urban Beijing: An Application of GeoShapley. Habitat International.
- Yang, A., Ai, J., & Arkolakis, C. (2025). A Geospatial Approach to Measuring Economic Activity (No. w33619). National Bureau of Economic Research (NBER).
- Guo, K., Tang, R., Pan, H., Zhang, D., Liu, Y., & Shi, Z. (2025). Activity Spaces in Multimodal Transportation Networks: A Nonlinear and Spatial Analysis Perspective. ISPRS International Journal of Geo-Information, 14(8), 281.
- Neto, J. B. P., Santos, N. F., & Orrico Filho, R. D. (2025). Paths to prosperity: How transport networks and income accessibility shape retail location. Journal of Transport Geography, 128, 104377.
- Nguyen, G. V., Van, C. P., Tran, V. N., Van, L. N., & Lee, G. (2025). Toward real-time high-resolution fluvial flood forecasting: A robust surrogate approach based on overland flow models. Environmental Modelling & Software, 106716.
- Chen, S., Qiu, Y., Xu, Y., Huang, J., & Ding, Z. (2025). Modeling and optimization of heat island networks based on machine learning and the perspective of spatial heterogeneity in metropolitan areas. Urban Climate, 63, 102592.
- Yang, J. T., Zhou, X. W., Chen, X., Yao, X., Li, M., Yin, M. H., ... & Zhao, J. L. (2025). Exploring the Main and Interactive Effects of Urban Morphology on Land Surface Temperature across Different Functional Zones: A Local GeoShapley Analysis Based on XAI. Sustainable Cities and Society, 106913.
- Yan, H., Wu, H., Su, X., Chu, J., & Askari, K. (2025). A new weighted framework for regional drought risk assessment and geographic explainable insights based on GeoShapley. Journal of Hydrology, 134460.
- Lyu, F., Chen, K., Olhnuud, A., Sun, X., & Gong, C. (2025). Understanding the Relationship Between Urban Green Infrastructure and PM2. 5 Based on an Explainable Machine Learning Model: Evidence From 288 Cities in China. Earth's Future, 13(11), e2025EF006861.
- Xiao, L., Wu, M., Weng, Q., & Liu, J. (2025). Exploring nonlinear and spatially varying relationships between built environment and the resilience of urban economic vibrancy under COVID-19. Applied Geography, 185, 103794.
- Pan, Q., Gao, S., Lin, S., & Liu, C. (2025). Exploring Spatio-Temporal Heterogeneity in Sustainable Development Drivers Using Explainable AI: Evidence from China. Finance Research Letters, 109018.

</details>
