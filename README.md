
![PyPI](https://img.shields.io/pypi/v/geoshapley)
![Pepy-Downloads](https://static.pepy.tech/badge/geoshapley)

# GeoShapley

<img src="https://github.com/Ziqi-Li/geoshapley/assets/5518908/b450b5b3-fd59-41d8-a64c-fb202f492302" width="500">



A game theory approach to measuring spatial effects from machine learning models. GeoShapley is built on Shapley value and Kernel SHAP estimator.

### Recent Updates
- 06/2026 v0.2.0 - GeoShapleyTreeExplainer for tree-based models that extends TreeSHAP. This would be highly efficient and preferred for tree-based models.
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

### Fast tree-path GeoShapley for tree models:

For supported tree regressors, `GeoShapleyTreeExplainer` computes an exact
tree-path-dependent GeoShapley decomposition without a background dataset. It
returns the same `GeoShapleyResults` object as `GeoShapleyExplainer`, so summary
plots, partial dependence plots, contribution bars, and `get_svc()` can be used
the same way.

```python
from geoshapley import GeoShapleyTreeExplainer
from xgboost import XGBRegressor

X_train, X_test, y_train, y_test = train_test_split(X_geo, y, random_state=1)

model = XGBRegressor(objective="reg:squarederror").fit(X_train.values, y_train)

# Spatial columns must still be the last columns in X_geo.
tree_explainer = GeoShapleyTreeExplainer(model, g=2)
tree_rslt = tree_explainer.explain(X_geo)

tree_rslt.summary_plot()
tree_rslt.partial_dependence_plots()
tree_svc = tree_rslt.get_svc(col=[0], coef_type="gwr", include_primary=True)
```

`GeoShapleyTreeExplainer` currently supports scikit-learn regression trees,
forest-style regressors, `GradientBoostingRegressor`, XGBoost sklearn models,
native `xgboost.Booster` objects, LightGBM sklearn models, native
`lightgbm.Booster` objects, and FLAML `AutoML` objects whose selected estimator
is a supported tree model. It uses tree path cover proportions to integrate out
missing players, analogous to TreeSHAP's tree-path-dependent setting. This is a
different value function from Kernel GeoShapley with a user-supplied background
dataset, though the two are often close when the background represents the tree
model's training distribution.

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


### Some recent papers that applied GeoShapley:

To date, GeoShapley has enabled more than 70 peer-reviewed empirical studies in geospatial explainable AI (XAI). These have been published in leading journals such as Journal of Transport Geography, Cities, Applied Geography, Landscape and Urban Planning, Sustainable Cities and Society, npj Urban Sustainability, Journal of Hydrology, Water Research, Ecological Indicators, Energy, Environmental Impact Assessment Review, Urban Climate, and Habitat International, Accident Analysis & Prevention, Environmental Modelling & Software, Journal of Environmental Management, Finance Research Letters, among others.

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
- Tan, G., Xing, Z., & Meng, F. (2025). Assessing and mapping the seasonal supply of recreational ecosystem services based on thermal comfort in Chongqing, China. Urban Climate, 64, 102661.
- Yang, F., Liu, M., Liu, S., Li, F., Liu, W., & Xu, C. (2025). Cross-media dynamics and prioritized risks of PFAS in textile-impacted environments: using geospatial machine learning. Environment International, 110008.
- He, Z., Chen, Y., Ning, Q., Lu, B., Xie, S., & Tang, S. (2025). Unraveling Nonlinear and Spatially Heterogeneous Impacts of Urban Pluvial Flooding Factors in a Hill-Basin City Using Geographically Explainable Artificial Intelligence: A Case Study of Changsha. Sustainability, 17(21), 9866.
- Bashar, T. J., Tao, R., Fernandes, C., & Jiao, Z. (2026). Evaluating spatial disparities in public EV charging infrastructure across the United States. Journal of Transport Geography, 130, 104507.
- Rui, J., & Gong, W. (2026). Paying lip service? An investigation into the spatial mismatch between younger and older adults' streetscape perceptual preference and visitation behavior. Cities, 171, 106750.
- Tang, Z., Rao, Y., & Fu, M. (2026). Synergistic Dynamics of the Thermal-Energy-Carbon Nexus in the Yangtze River Delta: Spatiotemporal Measurement, Mechanisms, and Spatial Econometric Analysis. Sustainable Cities and Society, 107125.
- Wang, K., Xi, C., Liu, X., Zheng, L., & Zhang, Y. (2026). Understanding process differences in the impact of built–natural environments on compound heat–flood risks through urban physical characteristics. Landscape and Urban Planning, 270, 105599.
- Xu, Y., Ma, R. L., Feng, Y. X., Zou, B., Li, S. X., Huang, W. T., ... & Peng, W. Q. (2026). A regional classification framework integrating AI and causal inference revealing the drivers of lake eutrophication in China. Water Research, 125746.
- Putra, I. G. B., Kuo, P. F., Susanta, F. F., Tedjo, B. H., & Lord, D. (2026). GeoShapley-based interpretation of older adult pedestrian fatal vs injury crash frequency in dense urban environments. Accident Analysis & Prevention, 230, 108450.
- Zhang, X., Hu, J., Xia, T., Mao, Y., Li, X., Hu, C., & Zhang, J. (2026). Cooling varies with green space characteristics: Unraveling nonlinear spatial heterogeneity in cooling effects of urban green spaces with geographic explainable AI. Environmental Impact Assessment Review, 119, 108401.
- Li, F., Zhang, Q., Xie, W., Shao, Z., & Li, J. (2026). Assessing the carbon reduction potential of municipal solid waste incineration at the county level in China. Resources, Environment and Sustainability, 100310.
- Nguyen, G. V., Van, C. P., Tran, V. N., Van, L. N., & Lee, G. (2025). Toward real-time high-resolution fluvial flood forecasting: A robust surrogate approach based on overland flow models. Environmental Modelling & Software, 106716.
- Hao, W., Xian, Z., Wei, Z., Xu, H., & Liu, K. (2026). The impact of multi-dimensional spatial characteristics on recreational walking experiences in nature-symbiotic city. Applied Geography, 188, 103865.
- Li, B., Zhang, Z., Wang, X., & Ren, S. (2026). Assessing stage-dependent impacts of environmental factors on public-perceived flood risk: Insights from a GeoXAI framework. Environmental Impact Assessment Review, 119, 108402.
- Xie, Y., Liu, R., & Fan, M. (2026). Evolutionary characteristics of spatial correlation network in energy-economy-environment (3E) coupled system and their driving factors, China. Energy, 140220.
- Pan, Q., Gao, S., Lin, S., & Liu, C. (2025). Exploring spatio-temporal heterogeneity in sustainable development drivers using explainable AI: Evidence from China. Finance Research Letters, 109018.
- Zhao, M., Lei, S., & Li, W. (2026). Incorporating urban thermal comfort into transit-oriented development (TOD) planning: Non-linear heterogeneous built environment effects. Sustainable Cities and Society.
- Lyu, F., Chen, K., Olhnuud, A., Sun, X., & Gong, C. (2025). Understanding the relationship between urban green infrastructure and PM2. 5 based on an explainable machine learning model: Evidence from 288 cities in China. Earth's Future, 13(11), e2025EF006861.
- Chen, L., & Liu, Y. (2026). Disentangling the invasion landscape: Spatially-explicit explainable machine learning reveals the heterogeneous drivers of Solenopsis invicta distribution in China. Journal of Environmental Management, 397, 128366.
- Ndagijimana, A., Nduwayezu, G., Lind, T., & Mansourian, A. (2025). Machine learning techniques to model child low height-for-age in the Northern Province of Rwanda: The role of climatological and environmental factors and their interactions. Clinical Epidemiology and Global Health, 102284.
- Chen, L., & Liu, Y. (2026). Deconstructing driver importance: A geospatial explainable AI approach to modeling pine wilt disease susceptibility in China. Forest Ecology and Management, 603, 123471.
- Lin, Y. F., & Lin, S. S. (2026). A spatially-coherent attribution framework for interpreting black-box tropical cyclone intensity forecasts. Stochastic Environmental Research and Risk Assessment, 40(4), 70.
- Kang, C. D. (2025). Examining the Nonlinear and Spatial Heterogeneity of Land Price Determinants in Seoul, Republic of Korea: An Application of GeoShapley. Journal of Real Estate Analysis, 11(3), 213-243.
- Luo, X., Tang, Z., Shen, Z., Xiao, L., Guo, H., & Li, H. (2026). Spatiotemporal Evolution, Nonlinear Driving Mechanisms, and Targeted Remediation of Cropland Fragmentation in the Sichuan Basin: A Temporally Enhanced Index and GeoXAI Approach. Land Degradation & Development.
- Ye, F., Wang, S., & Li, X. (2026). Attributing the urban heat island effect to urban land use patterns with integrated machine learning and GeoShapley method. Sustainable Cities and Society, 145, 107457. https://doi.org/10.1016/j.scs.2026.107457
- Yao, X., Zhang, J., Yang, J., Ye, B., & Zhu, Z. (2026). Unraveling the Impacts of Multidimensional Urban Influencing Factors on Daytime and Nighttime Thermal Environments across Urban Functional Zones using SDGSAT-1 Data and XGBoost-GeoShapley Analysis. Sustainable Cities and Society, 107508.
- Tang, Z., Rao, Y., & Fu, M. (2026). Synergistic Dynamics of the Thermal-Energy-Carbon Nexus in the Yangtze River Delta: Spatiotemporal Measurement, Mechanisms, and Spatial Econometric Analysis. Sustainable Cities and Society, 107125.
- Zhao, M., Lei, S., & Li, W. (2026). Incorporating urban thermal comfort into transit-oriented development (TOD) planning: Non-linear heterogeneous built environment effects. Sustainable Cities and Society.
- Luo, X., Bi, L., Chang, X., Wang, Q., Yang, D., & Li, S. (2026). Spatial heterogeneity and land use modulation of soil moisture–vapor pressure deficit–solar-induced fluorescence interactions in Henan, China: An integrated Random Forest–GeoShapley approach. Remote Sensing, 18(2), 235.
- Hao, W., Xian, Z., Wei, Z., Xu, H., & Liu, K. (2026). The impact of multi-dimensional spatial characteristics on recreational walking experiences in nature-symbiotic city. Applied Geography, 188, 103865.
- Chen, L., & Liu, Y. (2026). Deconstructing driver importance: A geospatial explainable AI approach to modeling pine wilt disease susceptibility in China. Forest Ecology and Management, 603, 123471.
- Dong, J., Yu, X., & Jia, G. (2026). Spatial heterogeneity and threshold effects of climate and human activities on ecological drought trends in northern China: Insights from a GeoShapley framework. GIScience & Remote Sensing.
- Chen, L., & Liu, Y. (2026). Disentangling the invasion landscape: Spatially-explicit explainable machine learning reveals the heterogeneous drivers of Solenopsis invicta distribution in China. Journal of Environmental Management, 397, 128366.
- Li, F., Zhang, Q., Xie, W., Shao, Z., & Li, J. (2026). Assessing the carbon reduction potential of municipal solid waste incineration at the county level in China. Resources, Environment and Sustainability, 100310.
- Aghazadeh, F., Ondrejicka, V., Sharifi, A., Aghaloo, K., Firozjaei, M. K., Rahimi, A., ... & Zhou, Y. (2026). Assessing urban thermal comfort: a multi-model analysis of European cities over two decades. npj Urban Sustainability.


</details>
