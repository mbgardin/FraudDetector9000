import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline

def make_lof_pipeline(preprocessor, n_neighbors: int = 35):
    """
    LOF with novelty=True so we can score val/test.
    Convention:
      - higher score => more anomalous
    """
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,   # critical: allows .predict/.score_samples on new data
        metric="minkowski"
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", lof),
    ])
    return pipe

def anomaly_score(pipe, X):
    """
    LOF score_samples: higher = more normal.
    We flip sign so higher = more anomalous.
    """
    normality = pipe.named_steps["model"].score_samples(
        pipe.named_steps["preprocess"].transform(X)
    )
    return (-normality).astype(float)