import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy.special
import itertools
import matplotlib.pyplot as plt
from math import factorial,ceil
from joblib import Parallel, delayed


class GeoShapleyExplainer:
    def __init__(self, predict_f, background=None, g=2):
        """
        Initialize the GeoShapleyExplainer.

        predict_f: The predict function of the model to be explained.
        background: The background data used for the explanation.
        g: The number of location features in the data (default is 2). For example, dataframe contains a pair of cooridnates (lat,long) g=2.
        """
        self.predict_f = predict_f
        self.background = background
        self.g = g
        self.n, self.M = background.shape
        

    def _kernel_geoshap_single(self, x, reference):
        """
        Calculate GeoShapley value for a single sample and a reference point in the background data

        x: current sample
        reference: a reference point in the background data
        """

    
        M = self.M
        Z = np.zeros((2**(M-1),M-1+M-2+1))

        #intercept
        Z[:,-1] = 1
    
        weights = np.zeros(2**(M-1))
    
        V = np.zeros((2**(M-1),M))
    
        for i in range(2**(M-1)):
            V[i,:] = reference

        #Mark 1 for each combination
        for i,s in enumerate(self._powerset(range(M-1))):
        
            s = list(s)
            Z[i,s] = 1
            V[i,s] = x[s]
        
            if (M-2) in s: #If location is in
                V[i, (M-1)] = x[(M-1)]
            
                if (len(s) > 1):
                    for j in s:
                      if j < (M-2):
                          Z[i, M-1+j] = 1
        
            weights[i] = self._shapley_kernel(M-1, len(s))
    
        y = self.predict_f(V).reshape(-1)


        #Solve WLS
        ZTw = np.dot(Z.T, np.diag(weights))
    
        phi = np.linalg.solve(np.dot(ZTw, Z), np.dot(ZTw, y))
    
        return phi


    def _kernel_geoshap_all(self, x):
        """
        Calculate GeoShapley value for a single sample and averaged over the background data

        x: current sample
        """
    
        n,M = self.background.shape
    
        # feature primary +
        # 2*geo_interaction to other features +
        # interaction + 
        # intercept
        phi = np.zeros(M + (M-2))
        
        for i in range(n):
            reference = self.background[i,:]
            phi = phi + self._kernel_geoshap_single(x, reference)
    
        phi = phi/n
        base_value = phi[-1]
        geoshap_values = phi[:-1]
    
        return base_value, geoshap_values



    def explain(self, X_geo, n_jobs=1):
        """
        Explain the data.

        X_geo: data to be explained

        return: A GeoShapleyResults object containing the results of the explanation.
        """
        
        self.X_geo = X_geo
        n,k = X_geo.shape

        geoshaps_total = np.zeros((n,(k-1+k-2)))
    
        # Parallel computation
        results = Parallel(n_jobs=n_jobs)(delayed(self._kernel_geoshap_all)(X_geo.values[i, :]) for i in tqdm(range(n)))

        # Extract results
        geoshaps_total = np.array([result[1] for result in results])
        base_value = results[0][0]  # Assuming base_value is same for all

        primary = geoshaps_total[:,:(k-2)]
        geo = geoshaps_total[:,(k-2)]
        geo_intera = geoshaps_total[:,(k-1):]
    
    
        return GeoShapleyResults(self, base_value, primary, geo, geo_intera)


    def _powerset(self, iterable):
        """
        Calculate possible coliation sets

        """
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


    def _shapley_kernel(self, M, s):
        """
        Calculate Shapley Kernel

        M: number of features
        s: number of features in the coalition
        """
        if s == 0 or s == M:
            return 100000000
        return (M-1)/(scipy.special.binom(M,s)*s*(M-s))

    


class GeoShapleyResults:
    def __init__(self, explainer, base_value, primary, geo, geo_intera):
        """
        Initializes the GeoShapleyResults.

        base_value: The base value
        primary: The primary global feature effects
        geo: The intrinsic location effect
        geo_intera: The interaction effects between location and other features
        X_geo: The data being explained

        """
        self.base_value = base_value
        self.primary = primary
        self.geo = geo
        self.geo_intera = geo_intera
        self.explainer = explainer
        self.predict_f = explainer.predict_f
        self.X_geo = explainer.X_geo
        self.g = explainer.g
        self.M = explainer.M
        self.background = explainer.background


    def get_svc(self, col, coef_type = "gwr", include_primary=False):
        """
        Calculate the spatial coefficient for each feature

        col: specify the column index to be calculated
        coef_type: 
            "raw": raw coefficient based on the ratio of interaction effect and mean removed feature value
            "gwr": coefficient based on GWR smoothing

        include_primary: whether to include the primary effect in the SVC

        """
    
        n,k = self.primary.shape
    
        params = np.zeros((n, k))
        params[:,:] = self.geo_intera 

        if include_primary:
            params[:,:] = params[:,:] + self.primary

        for j in col:
            if coef_type == "raw":
                params[:,j] = params[:,j] / (self.X_geo.values - self.X_geo.values.mean(axis=0))[:,j]

            if coef_type == "gwr":
                try:
                    import mgwr
                except ImportError:
                    print("Please install mgwr package (e.g., pip install mgwr)")
                
                coords = list(zip(self.X_geo.values[:,-2], self.X_geo.values[:,-1]))
                y = params[:,j].reshape(-1,1)
                X = (self.X_geo.values - self.X_geo.values.mean(axis=0))[:,j].reshape(-1,1)
                gwr_selector = mgwr.sel_bw.Sel_BW(coords, y, X,constant=False)
                gwr_bw = gwr_selector.search()
                gwr_model = mgwr.gwr.GWR(coords, y, X, gwr_bw,constant=False).fit()
                params[:,j] = gwr_model.params[:,0]
                print(gwr_model.R2)
    
        return params[:,col]
    

    def geoshap_to_shap(self):
        """
        Convert GeoShapley values to Shapley values.
        This will evenly redistribute the interaction effect evenly to a feature-location pair.

        """
        n,k = self.primary.shape
        params = np.zeros((n, k+1))
    
        params[:,:-1] = self.primary + self.geo_intera/2
        params[:,-1] = self.base_value + self.geo + np.sum(self.geo_intera/2,axis=1)
    
        return params


    def summary_plot(self, include_interaction=True, dpi=200, **kwargs):
        """
        Generate a SHAP-style summary plot of the GeoShapley values.
        
        include_interaction: whether to include the interaction effect in the summary plot
        dpi: figure dpi
        kwargs: other arguments passed to shap.summary_plot
        
        """

        try:
            import shap
        except ImportError:
            print("Please install shap package (e.g., pip install shap)")
        

        names = self.X_geo.iloc[:,:-2].copy()
        names["GEO"] = 0
    
        if include_interaction:
            total = np.hstack((self.primary, self.geo.reshape(-1,1), self.geo_intera))
            names[[name + " x GEO" for name in self.X_geo.columns[:-self.g]]] = self.X_geo.iloc[:,:-self.g].copy()
        else:
            total = self.geoshap_to_shap()
            
        plt.figure(dpi=dpi)
        shap.summary_plot(total, names, show=False, **kwargs)
    
        fig, ax = plt.gcf(), plt.gca()
        ax.set_xlabel("GeoShapley value (impact on model prediction)")


    def partial_dependence_plots(self, gam_curve=False, max_cols=3, figsize=(12, 12), dpi=200, **kwargs):
        """
        Plot partial dependence plots for each feature.

        gam_curve: whether to plot the smoothed GAM curve
        max_cols: maximum number of columns in the plot
        figsize: figure size
        dpi: figure dpi
        kwargs: other arguments passed to plt.scatter

        """

        k = self.primary.shape[1]

        if gam_curve:
            try:
                import pygam
            except ImportError:
                print("Please install pygam package (e.g., pip install pygam)")
    
        num_cols = min(k, max_cols)
        num_rows = ceil(k / num_cols)
        
        print(num_cols, num_rows)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
        axs = axs if num_rows > 1 else [axs]
        axs = axs.flatten()

        col_counter = 0
        for col in range(k):
            axs[col_counter].axhline(0, linestyle='--',color='black')

            if 's' not in kwargs:
                kwargs['s'] = 12
            if 'color' not in kwargs:
                kwargs['color'] = "#2196F3"
            if 'edgecolors' not in kwargs:
                kwargs['edgecolors'] = "white"
            if 'lw' not in kwargs:
                kwargs['lw'] = 0.3

            axs[col_counter].scatter(self.X_geo.iloc[:,col], self.primary[:,col],**kwargs)
        
            axs[col_counter].set_ylabel("GeoShapley Value")
            axs[col_counter].set_xlabel(self.X_geo.iloc[:,col].name)


            if gam_curve:
                lam = np.arange(40,201,20).reshape(-1,1)
                gam = pygam.LinearGAM(pygam.s(0),fit_intercept=False).gridsearch(self.X_geo.iloc[:,col].values.reshape(-1,1), 
                                                                                 self.primary[:,col].reshape(-1,1), lam=lam)
    
                for i, term in enumerate(gam.terms):
                    XX = gam.generate_X_grid(term=i)
                    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

                axs[col_counter].plot(XX,pdep, color="red",lw=2)

            col_counter += 1

        for i in range(col_counter, num_rows * num_cols):
            axs[i].axis('off')

        plt.tight_layout()


    def summary_statistics(self,include_interaction=True):
        """
        Calculates summary statistics for the GeoShapley values. 
        The table is ranked based on the mean absolute value of the GeoShapley values.

        include_interaction: whether to include the interaction effect in the summary statistics

        """
        cols = ["min","25%","50%","75%","max"]
        summary_table = pd.DataFrame(np.percentile(self.primary, [0,25,50,75,100],axis=0).T,columns=cols)
        summary_table.index = self.X_geo.columns[:-2]
        summary_table["mean"] = np.mean(self.primary,axis=0)
        summary_table["std"] = np.std(self.primary,axis=0)
        summary_table["abs. mean"] = np.mean(np.abs(self.primary),axis=0)

        summary_table.loc['GEO'] = np.append(np.percentile(self.geo, [0,25,50,75,100],axis=0).T, 
                                     [np.mean(self.geo), np.std(self.geo),
                                      np.mean(np.abs(self.geo))])
        

        if include_interaction:
            intera_summary_table = pd.DataFrame(np.percentile(self.geo_intera, [0,25,50,75,100],axis=0).T,columns=cols)
            intera_summary_table.index = self.X_geo.columns[:-2] + " x GEO"
            intera_summary_table["mean"] = np.mean(self.geo_intera,axis=0)
            intera_summary_table["std"] = np.std(self.geo_intera,axis=0)
            intera_summary_table["abs. mean"] = np.mean(np.abs(self.geo_intera),axis=0) 

            summary_table = pd.concat([summary_table, intera_summary_table], ignore_index=False)
        
        summary_table.sort_values(by=['abs. mean'],ascending=False,inplace=True)

        return summary_table

    def check_additivity(self,atol=1e-5):
        """
        Check if the seperate components of GeoShapley add up to the model prediction.

        """
        total = np.sum(self.primary,axis=1) + self.geo + np.sum(self.geo_intera,axis=1)
        
        print("Components add up to model prediction: ", 
              np.allclose(total+self.base_value, self.predict_f(self.X_geo).reshape(-1), atol=atol))