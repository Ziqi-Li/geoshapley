import scipy.special
import itertools
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import itertools
import matplotlib.pyplot as plt
from math import factorial,ceil

class GeoShapleyExplainer:
    def __init__(self, predict_f, background=None, g=2, exact=False, n_sampled_coalitions=1000):
        """
        Initialize the GeoShapleyExplainer.

        predict_f: The predict function of the model to be explained.
        background: The background data (numpy array) used for the explanation. It is suggested to use a small random sample (100 rows) or shap.kmeans(data,k=10).
        g: The number of location features in the data (default is 2). For example, feature set contains a pair of cooridnates (lat,long) g=2.
        n_sampled_coalitions: Number of random coalitions to sample for approximation (default is 400, which uses all coalitions).
        """
        self.predict_f = predict_f
        self.background = background
        self.g = g
        self.exact = exact
        self.n, self.M = background.shape
        self.k = self.M - self.g

        #For smaller # of features, force to be exact
        if self.k <= 8:
            self.exact = True
        
        if self.exact:
            self.n_coalitions = 2**(self.k+1)
        else:
            self.n_coalitions = min(2**(self.k+1), 2*(self.k+2) + self.k*(self.k+1) + n_sampled_coalitions)

        # Precompute powerset indices and shapley kernels
        self._precompute_powerset_and_kernel()
        
    def _precompute_powerset_and_kernel(self):
        """
        Precompute powerset indices and shapley kernels for faster computation
        """
        k = self.k
        M = k + 1  # Total number of features including location
        
        # Generate all possible coalitions (powerset)
        self.coalition_sizes = []
        self.coalition_indices = []
        
        # Store all 2^(k+1) coalitions
        for i, s in enumerate(self._powerset(range(k+1))):
            s = list(s)
            self.coalition_indices.append(s)
            self.coalition_sizes.append(len(s))
        
        # Convert to numpy arrays for vectorized operations
        self.coalition_sizes = np.array(self.coalition_sizes)
        
        # Precompute shapley kernels for all coalition sizes
        self.shapley_kernels = np.array([self._shapley_kernel(M, s) for s in self.coalition_sizes])
        
        # If n_coalitions is specified, sample the coalitions
        if self.n_coalitions is not None:
            self._sample_coalitions()
        else:
            # Use all coalitions
            self.sampled_indices = list(range(len(self.coalition_indices)))
            
            # Precompute Z matrix with all coalitions
            self._precompute_Z_matrix()
            
    def _sample_coalitions(self):
        """
        Sample coalitions based on Shapley weights for approximation
        """
        k = self.k
        coalition_count = len(self.coalition_indices)

        if self.exact:
            self.sampled_indices = list(range(coalition_count))
            self._precompute_Z_matrix()
            return
        
        # Identify important coalitions to always include
        must_have_indices = []
        
        # Empty coalition (always include)
        #null_idx = self.coalition_indices.index([])
        must_have_indices.append(0)
        
        # Full coalition (always include)
        must_have_indices.append(coalition_count - 1)

        # Single feature or two feature coalitions (always include)
        for i,coalition in enumerate(self.coalition_indices):
            if len(coalition) == 1:
                must_have_indices.append(i)
                must_have_indices.append(coalition_count - 1 - i)
            
        for i,coalition in enumerate(self.coalition_indices):
            if len(coalition) == 2:
                must_have_indices.append(i)
                must_have_indices.append(coalition_count - 1 - i)
        
        
        # Determine how many remaining coalitions to sample
        remaining_slots = self.n_coalitions - len(must_have_indices)
        
        if remaining_slots <= 0:
            # If we've already exceeded n_coalitions with must-have coalitions,
            # just get the first ones
            self.sampled_indices = must_have_indices[:self.n_coalitions]

        else:
            # Create sampling pool excluding must-have coalitions
            sampling_pool = [i for i in range(coalition_count) if i not in must_have_indices]
            
            # Get sampling weights for remaining coalitions
            weights = np.array([self.shapley_kernels[i] for i in sampling_pool])
            
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(sampling_pool)) / len(sampling_pool)

            # Sample remaining coalitions based on Shapley weights
            sampled_additional = np.random.choice(
                sampling_pool, 
                size=min(remaining_slots//2, len(sampling_pool)),
                replace=True,
                p=weights
            )

            paired = [coalition_count - 1 - i for i in sampled_additional]
            
            # Combine must-have and sampled coalitions
            self.sampled_indices = list(must_have_indices) + list(sampled_additional) + list(paired)
        
            #print(len(must_have_indices), len(sampled_additional), len(paired))
        
        # Sort indices for consistent ordering
        self.sampled_indices.sort()
        
        # Precompute Z matrix with sampled coalitions
        self._precompute_Z_matrix()
            
    def _precompute_Z_matrix(self):
        """
        Precompute Z matrix for selected coalitions
        """
        k = self.k
        # Get sampled coalitions
        sampled_coalitions = [self.coalition_indices[i] for i in self.sampled_indices]
        
        # Create Z matrix for sampled coalitions
        num_coalitions = len(sampled_coalitions)
        self.Z = np.zeros((num_coalitions, 2*k+1))
        
        # Fill Z matrix based on coalition structure
        for i, coalition in enumerate(sampled_coalitions):
            # Mark primary features
            if coalition:  # If coalition is not empty
                self.Z[i, coalition] = 1
                
            # Mark interaction features if location is included
            if k in coalition and len(coalition) > 1:
                for j in coalition:
                    if j < k:
                        self.Z[i, k+1+j] = 1
        
        # Get sampled shapley kernels
        self.sampled_kernels = np.array([self.shapley_kernels[i] for i in self.sampled_indices])

    def _vectorized_kernel_geoshap_single(self, x, reference):
        """
        Vectorized calculation of GeoShapley value for a single sample and reference point
        using sampled coalitions
        
        x: current sample
        reference: a reference point in the background data
        """
        k = self.k
        
        # Get sampled coalitions
        sampled_coalitions = [self.coalition_indices[i] for i in self.sampled_indices]
        num_coalitions = len(sampled_coalitions)
        
        # Create value matrix V for sampled coalitions
        V = np.tile(reference, (num_coalitions, 1))
        
        # For each coalition, replace feature values based on coalition membership
        for i, coalition in enumerate(sampled_coalitions):
            if coalition:  # If coalition is not empty
                V[i, coalition] = x[coalition]
            
            # If location feature is in coalition, replace location values
            if k in coalition:
                V[i, (k+1):] = x[(k+1):]
        
        # Predict outcomes for sampled coalition combinations
        y = self.predict_f(V).reshape(-1) - self.base_vale
        
        # Solve weighted least squares in vectorized form
        ZTw = self.Z.T * self.sampled_kernels
        phi = np.linalg.solve(np.dot(ZTw, self.Z), np.dot(ZTw, y))
        
        return phi

    def _vectorized_kernel_geoshap_all(self, x):
        """
        Vectorized calculation of GeoShapley values averaged over background dataset
        
        x: current sample
        """
        k = self.k
        
        # Initialize phi with zeros
        phi = np.zeros(2*k + 1)
        
        # Process each reference point with vectorized single computation
        for i in range(self.n):
            reference = self.background[i, :]
            phi += self._vectorized_kernel_geoshap_single(x, reference)
        
        # Average over background dataset
        phi = phi / self.n
        
        return phi
    
    def _vectorized_kernel_geoshap_all_batch(self, x):
        """
        Build a (C*n)×M matrix by tiling the background C times,
        then override each block of n rows per coalition.
        Instead of looping over single point in the background.
        """
        k, M, n = self.k, self.M, self.n
        # list of sampled coalitions
        coalitions = [self.coalition_indices[i] for i in self.sampled_indices]
        C = len(coalitions)

        # 1) Tile background into V_flat of shape (C*n, M)
        V_flat = np.tile(self.background, (C, 1))  # each block of n rows is one coalition

        # 2) For each coalition c, override the block rows c*n:(c+1)*n
        for c, coalition in enumerate(coalitions):
            rows = slice(c * n, (c + 1) * n)
            if coalition:  
                # x[coalition] has shape (len(coalition),),
                # assigning into V_flat[rows, coalition] (shape n×len(coalition)) will broadcast properly
                V_flat[rows, coalition] = x[coalition]
            if k in coalition:
                V_flat[rows, (k+1):] = x[(k+1):]

        # 3) One big predict call, then reshape to (C, n)
        y_flat = self.predict_f(V_flat)             # (C*n,)
        y_mat  = y_flat.reshape(C, n)

        # 4) Subtract base‐value and average over the n background points
        y_mean = (y_mat - self.base_vale).mean(axis=1)  # (C,)

        # 5) Single weighted least‐squares solve
        ZTW = self.Z.T * self.sampled_kernels           # (2k+1, C)
        A   = ZTW @ self.Z                              # (2k+1,2k+1)
        b   = ZTW @ y_mean                               # (2k+1,)
        phi = np.linalg.solve(A, b)

        return phi

    def explain(self, X_geo, n_jobs=-1):
        """
        Explain the data with vectorized computations.

        X_geo: pandas dataframe to be explained
        n_jobs: number of jobs for parallel computation (default is -1, using all available processors)

        return: A GeoShapleyResults object containing the results of the explanation.
        """
        self.X_geo = X_geo
        n, M = X_geo.shape
        k = M - self.g

        self.base_vale = np.mean(self.predict_f(self.background))
        
        # Parallel computation with vectorized method
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._vectorized_kernel_geoshap_all_batch)(X_geo.values[i, :]) 
            for i in tqdm(range(n))
        )

        # Extract results   
        geoshaps_total = np.vstack(results)
        base_value = self.base_vale

        primary = geoshaps_total[:, :k]
        geo = geoshaps_total[:, k]
        geo_intera = geoshaps_total[:, (k+1):]
    
        return GeoShapleyResults(self, base_value, primary, geo, geo_intera)

    def _powerset(self, iterable):
        """
        Calculate possible coalition sets
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
        return (M-1)/(scipy.special.binom(M, s)*s*(M-s))







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
                y = y - y.mean()
                X = (self.X_geo.values - self.X_geo.values.mean(axis=0))[:,j].reshape(-1,1)
                gwr_selector = mgwr.sel_bw.Sel_BW(coords, y, X, constant=False)
                gwr_bw = gwr_selector.search(bw_min=20)
                gwr_model = mgwr.gwr.GWR(coords, y, X, gwr_bw, constant=False).fit()
                params[:,j] = gwr_model.params[:,0]
    
        return params[:,col]
    

    def geoshap_to_shap(self):
        """
        Convert GeoShapley values to Shapley values.
        This will evenly redistribute the interaction effect to a feature-location pair.

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
                kwargs['s'] = 30
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
                lam = np.logspace(1, 7, 10).reshape(-1,1)
                gam = pygam.LinearGAM(pygam.s(0),fit_intercept=False).gridsearch(self.X_geo.iloc[:,col].values.reshape(-1,1), 
                                                                                 self.primary[:,col].reshape(-1,1), lam=lam)
    
                for i, term in enumerate(gam.terms):
                    XX = gam.generate_X_grid(term=i)
                    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

                axs[col_counter].plot(XX,pdep, color="red",lw=1)

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
