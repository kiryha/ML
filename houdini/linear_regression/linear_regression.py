"""
Goal: predict PolyReduce “Target Triangle Count” from simple geometry features.

We keep edge length fixed. Now we use TWO inputs:
- X_area (m^2)
- X_volume (m^3)

Linear model (no intercept):
y ≈ k_area * X_area + k_volume * X_volume
"""

import json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path


root_data = "C:/Users/kko8/OneDrive/projects/houdini_snippets/prod/3d/scenes/ML/Lab_1/data"
CSV = Path(f"{root_data}/triangle_density.csv")
df = pd.read_csv(CSV)


def plot_scatter_triarea(x, y, x_label, y_label, title, hue=None, size=None, logx=False, logy=False, bins=20):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- scatter ---
    if hue is None:
        axes[0].scatter(x, y)
    else:
        h = np.asarray(hue)
        sc = axes[0].scatter(x, y, c=h)
        cb = fig.colorbar(sc, ax=axes[0])
        cb.set_label("Volume")

    if size is not None:
        axes[0].collections[-1].set_sizes(np.asarray(size))

    if logx: axes[0].set_xscale("log")
    if logy: axes[0].set_yscale("log")
    axes[0].set_xlabel(x_label); axes[0].set_ylabel(y_label); axes[0].set_title(title)

    # --- hist x ---
    axes[1].hist(x, bins=bins)
    if logx: axes[1].set_xscale("log")
    axes[1].set_xlabel(x_label); axes[1].set_ylabel("Count")

    # --- hist y ---
    axes[2].hist(y, bins=bins)
    if logy: axes[2].set_xscale("log")
    axes[2].set_xlabel(y_label); axes[2].set_ylabel("Count")

    plt.tight_layout(); plt.show()


def train_model(d):
    """
    Train Linear Regression with two features and no intercept:
    y ≈ k_area * X_area + k_volume * X_volume
    """
    X = d[["X_area", "X_volume"]].to_numpy()
    y = d["y_total_prims"].to_numpy()
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    return model


def eval_on(df_split, tag, model):
    """
    Compute metrics comparing truth y vs prediction yhat on a split.
    """
    X = df_split[["X_area", "X_volume"]].to_numpy()
    y = df_split["y_total_prims"].to_numpy()

    k_area, k_volume = [float(c) for c in model.coef_]
    yhat = model.predict(X)

    r2  = r2_score(y, yhat)
    mae = mean_absolute_error(y, yhat)
    mape = float(np.mean(np.abs((y - yhat)/np.maximum(1e-9, y)))*100.0)

    print(f"[{tag}] k_area={k_area:.4f}  k_volume={k_volume:.4f}  R2={r2:.4f}  MAE={mae:.2f}  MAPE={mape:.2f}%  n={len(df_split)}")
    return r2, mae, mape, yhat


def train():
    # Split dataset: 80/10/10
    iters = df["iteration"].astype(int).to_numpy()
    m_train = (iters % 10) < 8
    m_val   = (iters % 10) == 8
    m_test  = (iters % 10) == 9
    df_train, df_validation, df_test = df[m_train], df[m_val], df[m_test]

    # Train
    model = train_model(df_train)

    # Eval
    r2_tr, mae_tr, mape_tr, _ = eval_on(df_train, "train", model)
    r2_va, mae_va, mape_va, _ = eval_on(df_validation, "val", model)
    r2_te, mae_te, mape_te, yhat_te = eval_on(df_test, "test", model)

    # Save model
    joblib.dump(model, f"{root_data}/triangle_density.joblib")


# Visualize (unchanged example view; area vs tris)
# plot_scatter_triarea(
#     df["X_area"], df["y_total_prims"],
#     "X_area", "y_total_prims",
#     "Total Primitives vs Area",
#     hue=df["X_volume"]
# )

train()
