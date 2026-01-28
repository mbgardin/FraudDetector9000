from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def make_logreg_pipeline(preprocessor):
    clf = LogisticRegression(
        max_iter=2000,
        class_weight=balanced,
        n_jobs=None
    )
    return Pipeline(steps=[
        (preprocess, preprocessor),
        (model, clf)
    ])
