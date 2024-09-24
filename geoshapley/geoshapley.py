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
        background: The background data (numpy array) used for the explanation.
        g: The number of location features in the data (default is 2). For example, feature set contains a pair of cooridnates (lat,long) g=2.
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

        #M = 4, g = 2, k = 2

        k = self.M - self.g
        M = self.M

        Z = np.zeros((2**(k+1),2*k+2))

        #intercept
        Z[:,-1] = 1
    
        weights = np.zeros(2**(k+1))
    
        V = np.zeros((2**(k+1),M))
    
        for i in range(2**(k+1)):
            V[i,:] = reference

        #Mark 1 for each combination
        for i,s in enumerate(self._powerset(range(k+1))):
        
            s = list(s)
            Z[i,s] = 1
            V[i,s] = x[s]
        
            if k in s: #If location is in
                V[i, (k+1):] = x[(k+1):]
            
                if (len(s) > 1): #mark interaction
                    for j in s:
                      if j < k:
                          Z[i, k+1+j] = 1
        
            weights[i] = self._shapley_kernel(k+1, len(s))
            #print("s:", s)
            #print("Z:", Z[i,:])

        y = self.predict_f(V).reshape(-1)

        #Solve WLS
        #ZTw = np.dot(Z.T, np.diag(weights))
        ZTw = Z.T * weights
        
        phi = np.linalg.solve(np.dot(ZTw, Z), np.dot(ZTw, y))
    
        return phi


    def _kernel_geoshap_all(self, x):
        """
        Calculate GeoShapley value for a single sample and averaged over the background data

        x: current sample
        """
    
        k = self.M - self.g
        n = self.n
    
        # feature primary +
        # 2*geo_interaction to other features +
        # interaction + 
        # intercept

        phi = np.zeros(k + k + 1 + 1)
        
        for i in range(n):
            reference = self.background[i,:]
            phi = phi + self._kernel_geoshap_single(x, reference)
    
        phi = phi/n
        base_value = phi[-1]
        geoshap_values = phi[:-1]
    
        return base_value, geoshap_values



    def explain(self, X_geo, n_jobs=-1):
        """
        Explain the data.

        X_geo: pandas dataframe to be explained
        n_jobs: number of jobs for parallel computation (default is -1, using all available processors)

        return: A GeoShapleyResults object containing the results of the explanation.
        """
        
        self.X_geo = X_geo
        n,M = X_geo.shape
        k = M - self.g

        geoshaps_total = np.zeros((n,(2*k + 1)))
    
        # Parallel computation
        results = Parallel(n_jobs=n_jobs)(delayed(self._kernel_geoshap_all)(X_geo.values[i, :]) for i in tqdm(range(n)))

        # Extract results
        geoshaps_total = np.array([result[1] for result in results])
        base_value = results[0][0]  # Assuming base_value is same for all

        primary = geoshaps_total[:,:k]
        geo = geoshaps_total[:,k]
        geo_intera = geoshaps_total[:,(k+1):]
    
    
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


    def get_svc(self, col, coef_type = "gwr", include_primary=False, coords=None):
        """
        Calculate the spatial (location-spefific) coefficient for each feature

        col: specify the column index to be calculated
        coef_type: 
            "raw": raw coefficient based on the ratio of interaction effect and mean removed feature value. 
                   May result in extreme values.
            "gwr": coefficient based on GWR smoothing. Requires mgwr package.
        
        include_primary: whether to include the primary effect in the spatial coefficient
        coords: a numpy array of the coordinates of the data. If not provided, the last two columns of the data will be used as coordinates.
        
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
                
                if coords is None: #Assuming the last two columns are the coordinates
                    coords = np.array(list(zip(self.X_geo.values[:,-2], self.X_geo.values[:,-1])))

                y = params[:,j].reshape(-1,1)
                X = (self.X_geo.values - self.X_geo.values.mean(axis=0))[:,j].reshape(-1,1)
                gwr_selector = mgwr.sel_bw.Sel_BW(coords, y, X)
                gwr_bw = gwr_selector.search(bw_min=20)
                gwr_model = mgwr.gwr.GWR(coords, y, X, gwr_bw).fit()
                params[:,j] = gwr_model.params[:,1]
    
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
        

        names = self.X_geo.iloc[:,:-self.g].copy()
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


    def partial_dependence_plots(self, gam_curve=False, max_cols=3, figsize=None, dpi=200, **kwargs):
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

        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
        axs = axs if num_rows > 1 else np.array([axs])
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
                lam = np.logspace(2, 7, 5).reshape(-1,1)
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
        summary_table.index = self.X_geo.columns[:-self.g]
        summary_table["mean"] = np.mean(self.primary,axis=0)
        summary_table["std"] = np.std(self.primary,axis=0)
        summary_table["abs. mean"] = np.mean(np.abs(self.primary),axis=0)

        summary_table.loc['GEO'] = np.append(np.percentile(self.geo, [0,25,50,75,100],axis=0).T, 
                                     [np.mean(self.geo), np.std(self.geo),
                                      np.mean(np.abs(self.geo))])
        

        if include_interaction:
            intera_summary_table = pd.DataFrame(np.percentile(self.geo_intera, [0,25,50,75,100],axis=0).T,columns=cols)
            intera_summary_table.index = self.X_geo.columns[:-self.g] + " x GEO"
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
