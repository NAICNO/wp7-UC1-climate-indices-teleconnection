# models.py
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

class LRDropoutPSO(LinearRegression):
    def __init__(self, max_diff=0.25, n_jobs=-1, max_iter=1000, population_size=100, inertia=0.5, cognitive=2.0, social=2.0, dropout_rate=0.2, *args, **kwargs):
        """
        Initialize the LRDropoutPSO class.

        Parameters:
        - max_diff: float, default=0.25
            Maximum difference allowed between weights in the PSO algorithm.
        - n_jobs: int, default=-1
            Number of parallel jobs to run. -1 means using all processors.
        - max_iter: int, default=1000
            Maximum number of iterations for the PSO algorithm.
        - population_size: int, default=100
            Number of particles in the PSO algorithm.
        - inertia: float, default=0.5
            Inertia weight for the PSO algorithm.
        - cognitive: float, default=2.0
            Cognitive weight for the PSO algorithm.
        - social: float, default=2.0
            Social weight for the PSO algorithm.
        - dropout_rate: float, default=0.1
            Dropout rate for uncertainty estimation.
        - *args, **kwargs: additional arguments to be passed to the LinearRegression class.
        """
        self.max_diff = max_diff
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.max_iter = max_iter
        self.population_size = population_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.dropout_rate = dropout_rate
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        """
        Fit the LRDropoutPSO model to the training data with dropout.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            Training data.
        - y: array-like of shape (n_samples,)
            Target values.

        Returns:
        - self: fitted estimator.
        """
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Define the loss function (Sum of squared errors)
        def loss_function(coefficients):
            errors = X_scaled.dot(coefficients) - y
            return np.dot(errors, errors)  # Sum of squared errors

        # Define the constraint: no weight can be more than max_diff% different from the others
        def constraint(coefficients):
            max_coef = np.max(np.abs(coefficients))
            min_coef = np.min(np.abs(coefficients))
            if max_coef > 0:
                return self.max_diff * max_coef - (max_coef - min_coef)
            return 0

        # Initialize PSO parameters
        dim = X.shape[1]
        lb = -10 * np.ones(dim)
        ub = 10 * np.ones(dim)

        # Initialize particles
        particles = np.random.uniform(lb, ub, (self.population_size, dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([loss_function(p) for p in particles])

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        def update_particle(i):
            """
            Update a particle in the PSO algorithm.

            Parameters:
            - i: int
                Index of the particle to update.
            """
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (self.inertia * velocities[i] +
                             self.cognitive * r1 * (personal_best_positions[i] - particles[i]) +
                             self.social * r2 * (global_best_position - particles[i]))

            # Apply dropout by randomly zeroing out some velocities
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=velocities[i].shape)
            velocities[i] *= dropout_mask

            particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
            
            if constraint(particles[i]) >= 0:
                score = loss_function(particles[i])
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_scores[i] = score

        # PSO main loop
        for _ in range(self.max_iter):
            Parallel(n_jobs=self.n_jobs)(delayed(update_particle)(i) for i in range(self.population_size))
            global_best_index = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index]

        self.coef_ = global_best_position
        self.intercept_ = np.mean(y) - np.dot(np.mean(X_scaled, axis=0), self.coef_)

        return self
    
    def set_coef(self, logged_feature_importances, X, y):
        """
        Set the coefficients of the LRDropoutPSO model.

        Parameters:
        - logged_feature_importances: array-like of shape (n_features,)
            Logged feature importances.
        - X: array-like of shape (n_samples, n_features)
            Training data.
        - y: array-like of shape (n_samples,)
            Target values.
        """
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.coef_ = np.array(logged_feature_importances)        
        self.intercept_ = np.mean(y) - np.dot(np.mean(X_scaled, axis=0), self.coef_)

    def predict(self, X, with_uncertainty=False, n_iter=100):
        """
        Predict using the LRDropoutPSO model with dropout to estimate uncertainty.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            Samples.
        - n_iter: int, default=100
            Number of iterations for prediction with dropout.

        Returns:
        - y_pred: array-like of shape (n_samples,)
            Predicted values.
        - y_std: array-like of shape (n_samples,)
            Standard deviation of predictions for uncertainty estimation.
        """
        X_scaled = self.scaler_.transform(X)
        predictions = np.zeros((n_iter, X_scaled.shape[0]))

        for i in range(n_iter):
            # Apply dropout by randomly zeroing out some coefficients
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.coef_.shape)
            coef_dropout = self.coef_ * dropout_mask
            predictions[i] = np.dot(X_scaled, coef_dropout) + self.intercept_

        y_pred = predictions.mean(axis=0)
        y_std = predictions.std(axis=0)
        if(with_uncertainty):
            return y_pred, y_std
        else:
            return y_pred
        

    def predict_single(self, X):
        """
        Predict using the LRDropoutPSO model without dropout.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            Samples.

        Returns:
        - y_pred: array-like of shape (n_samples,)
            Predicted values.
        """
        X_scaled = self.scaler_.transform(X)
        return np.dot(X_scaled, self.coef_) + self.intercept_

class LRforcedPSO(LinearRegression):
    def __init__(self, max_diff=0.25, n_jobs=-1, max_iter=1000, population_size=100, inertia=0.5, cognitive=2.0, social=2.0, *args, **kwargs):
        """
        Initialize the LRforcedPSO class.

        Parameters:
        - max_diff: float, default=0.25
            Maximum difference allowed between weights in the PSO algorithm.
        - n_jobs: int, default=-1
            Number of parallel jobs to run. -1 means using all processors.
        - max_iter: int, default=1000
            Maximum number of iterations for the PSO algorithm.
        - population_size: int, default=100
            Number of particles in the PSO algorithm.
        - inertia: float, default=0.5
            Inertia weight for the PSO algorithm.
        - cognitive: float, default=2.0
            Cognitive weight for the PSO algorithm.
        - social: float, default=2.0
            Social weight for the PSO algorithm.
        - *args, **kwargs: additional arguments to be passed to the LinearRegression class.
        """
        self.max_diff = max_diff
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.max_iter = max_iter
        self.population_size = population_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        """
        Fit the LRforcedPSO model to the training data.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            Training data.
        - y: array-like of shape (n_samples,)
            Target values.

        Returns:
        - self: fitted estimator.
        """
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Define the loss function (Sum of squared errors)
        def loss_function(coefficients):
            errors = X_scaled.dot(coefficients) - y
            return np.dot(errors, errors)  # Sum of squared errors

        # Define the constraint: no weight can be more than max_diff% different from the others
        def constraint(coefficients):
            max_coef = np.max(np.abs(coefficients))
            min_coef = np.min(np.abs(coefficients))
            if max_coef > 0:
                return self.max_diff * max_coef - (max_coef - min_coef)
            return 0

        # Initialize PSO parameters
        dim = X.shape[1]
        lb = -10 * np.ones(dim)
        ub = 10 * np.ones(dim)

        # Initialize particles
        particles = np.random.uniform(lb, ub, (self.population_size, dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([loss_function(p) for p in particles])

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        def update_particle(i):
            """
            Update a particle in the PSO algorithm.

            Parameters:
            - i: int
                Index of the particle to update.
            """
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (self.inertia * velocities[i] +
                             self.cognitive * r1 * (personal_best_positions[i] - particles[i]) +
                             self.social * r2 * (global_best_position - particles[i]))
            particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
            
            if constraint(particles[i]) >= 0:
                score = loss_function(particles[i])
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_scores[i] = score

        # PSO main loop
        for _ in range(self.max_iter):
            Parallel(n_jobs=self.n_jobs)(delayed(update_particle)(i) for i in range(self.population_size))
            global_best_index = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index]

        self.coef_ = global_best_position
        self.intercept_ = np.mean(y) - np.dot(np.mean(X_scaled, axis=0), self.coef_)

        return self
    
    def set_coef(self, logged_feature_importances, X, y):
        """
        Set the coefficients of the LRforcedPSO model.

        Parameters:
        - logged_feature_importances: array-like of shape (n_features,)
            Logged feature importances.
        - X: array-like of shape (n_samples, n_features)
            Training data.
        - y: array-like of shape (n_samples,)
            Target values.
        """
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.coef_ = np.array(logged_feature_importances)        
        self.intercept_ = np.mean(y) - np.dot(np.mean(X_scaled, axis=0), self.coef_)

    def predict(self, X):
        """
        Predict using the LRforcedPSO model.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            Samples.

        Returns:
        - y_pred: array-like of shape (n_samples,)
            Predicted values.
        """
        X_scaled = self.scaler_.transform(X)
        return np.dot(X_scaled, self.coef_) + self.intercept_    

class ModelSelector:
    def __init__(self, modelname):
        """
        Initialize the ModelSelector class.

        Parameters:
        - modelname: str
            Name of the model to select.
        """
        self.modelname = modelname
        self.model = self._create_model()
    
    def _create_model(self):
        """
        Create the selected model.

        Returns:
        - model: estimator 
            Selected model.
        """
        if self.modelname == "LinearRegression":
            return LinearRegression()
        elif self.modelname == "LRDropoutPSO":
            return LRDropoutPSO(max_diff=0.25, dropout_rate=0.1)
        elif self.modelname == "LRDropoutPSO_50percent":
            return LRDropoutPSO(max_diff=0.5, dropout_rate=0.1)
        elif self.modelname == "LRforcedPSO_10percent":
            return LRforcedPSO(max_diff=0.10)
        elif self.modelname == "LRforcedPSO":
            return LRforcedPSO(max_diff=0.25)
        elif self.modelname == "LRforcedPSO_50percent":
            return LRforcedPSO(max_diff=0.5)
        elif self.modelname == "LRforcedPSO_75percent":
            return LRforcedPSO(max_diff=0.75)
        elif self.modelname == "MLPRegressor":
            return MLPRegressor()
        elif self.modelname == "RandomForestRegressor":
            return RandomForestRegressor()
        elif self.modelname == "XGBoost":
            # Configure XGBoost to use GPU
            return xgb.XGBRegressor(
                tree_method='gpu_hist',  # Use GPU for histogram-based algorithms
                gpu_id=0,  # Use the first GPU (if you have multiple GPUs, you can change this)
                predictor='gpu_predictor'  # Use GPU for prediction
            )
        else:
            raise ValueError("Invalid model name")
    
    def __getattr__(self, attr):
        """
        Delegate attribute access to the underlying model.
        """
        return getattr(self.model, attr)

    def feature_importances(self):
        """
        Get the feature importances of the selected model.

        Returns:
        - importances: array-like
            Feature importances.
        """
        if self.modelname in ["LinearRegression", "LRforcedPSO", "LRDropoutPSO",
                              "LRforcedPSO_10percent", "LRforcedPSO_50percent",
                              "LRforcedPSO_75percent", "LRDropoutPSO_50percent"]:
            return np.abs(self.model.coef_)
        elif self.modelname == "RandomForestRegressor":
            return self.model.feature_importances_
        elif self.modelname == "XGBoost":
            return self.model.feature_importances_
        elif self.modelname == "MLPRegressor":
            # MLPRegressor doesn't have feature_importances_, use input layer weights
            return np.abs(self.model.coefs_[0]).mean(axis=1)
        else:
            raise ValueError(f"Invalid model name: {self.modelname}")

