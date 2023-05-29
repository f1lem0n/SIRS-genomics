"""Utility functions"""

from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, roc_auc_score, roc_curve

import plotly.express as px

# global variables
acc_scores = {}
fprs = {}
tprs = {}
auc_scores = {}


def evaluate(model: object, X, y, model_name: str) -> None:
    # confusion matrix
    tick_text = ["0", "1"]
    fig = px.imshow(
        confusion_matrix(model.predict(X), y),
        x=tick_text, y=tick_text
    )
    fig.update_layout(dict(
        title=f"Confusion Matrix of {model_name}",
        yaxis_title="True label",
        xaxis_title="Predicted label",
        coloraxis_colorbar=dict(title="Number classified")
    ))
    fig.show()
    # classification report
    print("CLASSIFICATION REPORT:")
    print(classification_report(model.predict(X), y, zero_division=True))
    acc_scores[model_name] = accuracy_score(model.predict(X), y)
    fprs[model_name] = roc_curve(y, model.predict_proba(X)[:, 1])[0]
    tprs[model_name] = roc_curve(y, model.predict_proba(X)[:, 1])[1]
    auc_scores[model_name] = roc_auc_score(y, model.predict_proba(X)[:, 1])
