# fiber_predictor/svr_hog/models.py
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA

class FiberPredictor:
    def __init__(self, scaler_type="Standard", scaler_params=None, pca_params=None, regressor_type="SVR", regressor_params=None):
        """
        Initialize the FiberPredictor model pipeline.
        
        Parameters:
        - scaler_type (str): Type of scaler to use ('Standard', 'MinMax', 'Normalizer').
        - scaler_params (dict): Parameters for the scaler.
        - pca_params (dict): Parameters for PCA (e.g., {'n_components': 0.95}).
        - regressor_type (str): Type of regressor to use ('SVR', 'LinearRegression', 'Ridge').
        - regressor_params (dict): Parameters for the regressor.
        """
        self.scaler_type = scaler_type
        self.scaler_params = scaler_params if scaler_params is not None else {}
        self.pca_params = pca_params if pca_params is not None else {}
        self.regressor_type = regressor_type
        self.regressor_params = regressor_params if regressor_params is not None else {}

        # Build the model pipeline
        self.model = self.build_model()

    def get_scaler(self):
        if self.scaler_type == "Standard":
            return StandardScaler(**self.scaler_params)
        elif self.scaler_type == "MinMax":
            return MinMaxScaler(**self.scaler_params)
        elif self.scaler_type == "Normalizer":
            return Normalizer(**self.scaler_params)
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def get_regressor(self):
        if self.regressor_type == "SVR":
            return SVR(**self.regressor_params)
        elif self.regressor_type == "LinearRegression":
            return LinearRegression(**self.regressor_params)
        elif self.regressor_type == "Ridge":
            return Ridge(**self.regressor_params)
        else:
            raise ValueError(f"Unknown regressor type: {self.regressor_type}")

    def build_model(self):
        # Initialize the scaler
        scaler = self.get_scaler()

        # Initialize PCA
        pca = PCA(**self.pca_params) if self.pca_params else None

        # Initialize the regressor
        regressor = self.get_regressor()

        # Create the pipeline steps
        steps = [("scaler", scaler)]
        if pca is not None:
            steps.append(("pca", pca))
        steps.append(("regressor", regressor))

        # Create and return the model pipeline
        return Pipeline(steps)

    def fit(self, X, y):
        """Fit the model to the training data."""
        return self.model.fit(X, y)

    def predict(self, X):
        """Make predictions using the fitted model."""
        return self.model.predict(X)


# Example usage of the new FiberPredictor:
if __name__ == "__main__":
    # Custom parameters for each component
    scaler_params = {"with_mean": True, "with_std": True}
    pca_params = {"n_components": 0.95}
    regressor_params = {"kernel": "rbf", "C": 10, "gamma": 0.1}

    # Initialize the FiberPredictor with SVR and StandardScaler
    fiber_predictor_svr = FiberPredictor(
        scaler_type="Standard",
        scaler_params=scaler_params,
        pca_params=pca_params,
        regressor_type="SVR",
        regressor_params=regressor_params
    )

    # Build the SVR model
    svr_model = fiber_predictor_svr.build_model()
    print(svr_model)

    # Initialize the FiberPredictor with Linear Regression and Cosine Transformation
    fiber_predictor_lr_cosine = FiberPredictor(
        scaler_type="Cosine",
        pca_params=pca_params,
        regressor_type="LinearRegression"
    )

    # Build the Linear Regression model with Cosine normalization
    lr_cosine_model = fiber_predictor_lr_cosine.build_model()
    print(lr_cosine_model)

    # Initialize the FiberPredictor with Random Forest Regressor and Min-Max Scaling
    rf_params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    fiber_predictor_rf = FiberPredictor(
        scaler_type="MinMax",
        pca_params=pca_params,
        regressor_type="RandomForest",
        regressor_params=rf_params
    )

    # Build the Random Forest model with MinMaxScaler
    rf_model = fiber_predictor_rf.build_model()
    print(rf_model)
