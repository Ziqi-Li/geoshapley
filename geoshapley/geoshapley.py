import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.special
import itertools
import matplotlib.pyplot as plt
from math import factorial


class GeoShapleyExplainer:
    def __init__(self, predict_f, background=None):
        """
        Initialize the GeoShapleyExplainer.

        predict_f: The predict function of the model to be explained.
        background: The background data used for the explanation.
        """
        self.predict_f = predict_f
        self.background = background
        self.n, self.M = background.shape
        

    def kernel_geoshap_single(self, x, reference):
        """
        Calculate GeoShapley value for a single sample and a reference point in the background data

        :param x: current sample
        :param reference: a reference point in the background data
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
        for i,s in enumerate(self.powerset(range(M-1))):
        
            s = list(s)
            Z[i,s] = 1
            V[i,s] = x[s]
        
            if (M-2) in s: #If location is in
                V[i, (M-1)] = x[(M-1)]
            
                if (len(s) > 1):
                    for j in s:
                      if j < (M-2):
                          Z[i, M-1+j] = 1
        
            weights[i] = self.shapley_kernel(M-1, len(s))
    
        y = self.predict_f(V).reshape(-1)


        #Solve WLS
        ZTw = np.dot(Z.T, np.diag(weights))
    
        phi = np.linalg.solve(np.dot(ZTw, Z), np.dot(ZTw, y))
    
        return phi


    def kernel_geoshap_all(self, x):
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
            phi = phi + self.kernel_geoshap_single(x, reference)
    
        phi = phi/n
        base_value = phi[-1]
        geoshap_values = phi[:-1]
    
        return base_value, geoshap_values



    def explain(self, X_geo):
        """
        Explain the entire data.
        X_geo: data to be explained

        :return: A GeoShapleyResults object containing the results of the explanation.

        """
        
        self.X_geo = X_geo
        n,k = X_geo.shape

        geoshaps_total = np.zeros((n,(k-1+k-2)))
    
        for i in tqdm(range(n), leave=False):
            x = X_geo.values[i,:]
            base_value, geoshap_values = self.kernel_geoshap_all(x)
            geoshaps_total[i,:] = geoshap_values
        
        primary = geoshaps_total[:,:(k-2)]
        geo = geoshaps_total[:,(k-2)]
        geo_intera = geoshaps_total[:,(k-1):]
    
    
        return GeoShapleyResults(self, base_value, primary, geo, geo_intera)


    def powerset(self, iterable):
        """
        Calculate possible coliation sets

        """
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

    def shapley_kernel(self, M, s):
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

        :param base_value: The base value
        :param primary: The primary global effects
        :param geo: The intrinsic location effect
        :param geo_intera: The interaction effects between location and other features
        :param X_geo: The data explained

        """
        self.base_value = base_value
        self.primary = primary
        self.geo = geo
        self.geo_intera = geo_intera
        self.explainer = explainer
        self.predict_f = explainer.predict_f
        self.X_geo = explainer.X_geo
        self.background = explainer.background


    def attribute_to_X(self, svc_coef_col = []):
        """
        Cal
        
        """
    
        n,k = self.primary.shape
    
        params = np.zeros((n, k+1))
        params[:,:-1] = self.primary + self.geo_intera
        params[:,-1] = self.base_value + self.geo
    
        for j in svc_coef_col:
            params[:,j] = params[:,j] / (self.X_geo.values-self.background.mean(axis=0))[:,j]
    
        return np.roll(params, 1,axis=1)
    

    def geoshap_to_shap(self):



        n,k = self.primary.shape
        params = np.zeros((n, k))
    
        params[:,:-1] = self.primary + self.geo_intera/2
        params[:,-1] = self.base_value + self.geo + np.sum(self.geo_intera/2,axis=0)
    
        return params

    def summary_plot(self, include_interaction=True, dpi=200):
        """
        Generate a SHAP-style summary plot of the GeoShapley values.
        
        """

        try:
            import shap
        except ImportError:
            print("Please install shap package")
            return None
        

        temp = self.X_geo.iloc[:,:-2].copy()
        temp["GEO"] = 0
    
        if include_interaction:
            total = np.hstack((self.primary,self.geo.reshape(-1,1),self.geo_intera))
        
            temp[[name + " x GEO" for name in self.X_geo.columns[:-2]]] = self.X_geo.iloc[:,:-2].copy()
        else:
            total = np.hstack((primary,geo.reshape(-1,1)))
        
        #total_exp = (10**(total) - 1)*100
    
        plt.figure(dpi=dpi)
        shap.summary_plot(total,temp,show=False)
    
        fig, ax = plt.gcf(), plt.gca()
        ax.set_xlabel("GeoShapley value (impact on model prediction)")


    def summary_statistics(self):
        """
        Calculates summary statistics for the GeoShapley values.

        """

        #To-do
        pass

    def check_additivity(self,atol=1e-5):
        """
        Visualizes the results in an appropriate format, such as maps or graphs.

        :return: Visualization of the Shapley value results.
        """
        total = np.sum(self.primary,axis=1) + self.geo + np.sum(self.geo_intera,axis=1)
        
        print("Components add up to model prediction: ", 
              np.allclose(total+self.base_value, self.predict_f(self.X_geo).reshape(-1), atol=atol))