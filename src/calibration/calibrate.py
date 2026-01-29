import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator

def fit_calibrator_prefit(fitted_estimator, X_cal, y_cal, method: str):
    """
    scikit-learn compatible 'prefit' calibration using FrozenEstimator.
    Base estimator is already fit (train). Calibrator is fit on (X_cal, y_cal).
    """
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be 'sigmoid' or 'isotonic'")

    frozen = FrozenEstimator(fitted_estimator)
    cal = CalibratedClassifierCV(estimator=frozen, method=method, cv=5)
    cal.fit(X_cal, y_cal)
    return cal

def get_calibration_curve(y_true, y_prob, n_bins: int = 10):
    """
    Returns (fraction_of_positives, mean_predicted_value)
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return frac_pos, mean_pred

def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))