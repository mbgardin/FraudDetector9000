import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

def make_isoforest_pipeline(preprocessor, random_state: int = 42):
    """
    Returns a pipeline that outputs anomaly scores.
    Convention:
      - higher score => more anomalous (more likely fraud)
    """
    iso = IsolationForest(
        n_estimators=300,
        contamination="auto",   # we won't rely on this; we threshold ourselves
        random_state=random_state,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", iso),
    ])
    return pipe

def anomaly_score(pipe, X):
    """
    IsolationForest has score_samples where higher = more normal.
    We flip sign so higher = more anomalous.
    """
    normality = pipe.named_steps["model"].score_samples(pipe.named_steps["preprocess"].transform(X))
    return (-normality).astype(float)