import json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path


root_data = "C:/Users/kko8/OneDrive/projects/houdini_snippets/prod/3d/scenes/ML/Lab_1/data"
CSV = Path(f"{root_data}/triangles_per_area.csv")


# 1) Load
df = pd.read_csv(CSV)


def print_df_info(df):
    # 2) Quick shape & columns
    print(df.shape)           # (rows, cols)
    print(df.columns.tolist())

    # 3) Peek at first rows
    print(df.head(10))

    # 4) Data types & nulls
    print(df.info())
    print(df.isna().sum())    # should be zeros

    # 5) Basic stats
    print(df.describe())


def plot_scatter(x, y, x_label, y_label, title):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scatter: volume vs mass
    axes[0].scatter(x, y)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].set_title(title)
    
    # Histograms (same figure)
    axes[1].hist(x, bins=20)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Count")
    
    axes[2].hist(y, bins=20)
    axes[2].set_xlabel(y_label)
    axes[2].set_ylabel("Count")
    
    plt.tight_layout()
    plt.show()


def plot_scatter_triarea(x, y, x_label, y_label, title, hue=None, size=None, logx=False, logy=False, bins=20):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- scatter ---
    if hue is None:
        axes[0].scatter(x, y)
    else:
        # normalize hue to [0,1] and color by it; add a colorbar
        h = np.asarray(hue)
        sc = axes[0].scatter(x, y, c=h)
        cb = fig.colorbar(sc, ax=axes[0])
        cb.set_label("iteration")

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


# Iteration 2
# Goal: predict a PolyReduce “Target Triangle Count” from the current area.
def train_model(d):
    """
    Train the Linear Regression model.
    If we have 0 triangles, then we will have 0 area, hence we will have 0 slope and graph will pass through origin,
    so we need to fit_intercept=False
    """

    X = d[["X_area"]].to_numpy()
    y = d["y_total_prims"].to_numpy()
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    return model


def eval_on(df, tag):
    """
    Compute metrics comparing truth "y" vs prediction "yhat"

    k=58.7147 R2=0.9965 MAE=675.63 MAPE=4.59%

    R2: R-squared score, measures how much of the variation in y the model explains 
    Range ~[0,1] (higher is better). 1.0 is a perfect line.

    MAE (mean absolute error): average absolute difference |y − yhat|, in triangles. Easy to read in real units.
    On average, predictions are off by ~676 triangles. Whether that’s “big” depends on typical counts

    MAPE (mean absolute percentage error): average percentage error |y − yhat| / y, in %.
    Average relative error ~4.6%. This is small—good realism with the noise you injected.
    """

    X = df[["X_area"]].to_numpy()
    y = df["y_total_prims"].to_numpy()
    yhat = model.predict(X)
    r2  = r2_score(y, yhat)
    mae = mean_absolute_error(y, yhat)
    mape = float(np.mean(np.abs((y - yhat)/np.maximum(1e-9, y)))*100.0)
    print(f"[{tag}] k={k:.4f}  R2={r2:.4f}  MAE={mae:.2f}  MAPE={mape:.2f}%  n={len(df)}")\

    return r2, mae, mape, yhat


# Visualize data
# plot_scatter_triarea(df["X_area"], df["y_total_prims"], "X_area", "y_total_prims", "Total Primitives vs Area",hue=df["iteration"])


# Split dataset: 80/10/10 
iters = df["iteration"].astype(int).to_numpy()
m_train = (iters % 10) < 8
m_val   = (iters % 10) == 8
m_test  = (iters % 10) == 9
df_train, df_validation, df_test = df[m_train], df[m_val], df[m_test]

# Train
model = train_model(df_train)
k = float(model.coef_[0])   # triangles per m²

r2_tr, mae_tr, mape_tr, _ = eval_on(df_train, "train")
r2_va, mae_va, mape_va, _ = eval_on(df_validation, "val")
r2_te, mae_te, mape_te, yhat_te = eval_on(df_test, "test")

# Save model
joblib.dump(model, f"{root_data}/triangles_per_area.joblib")
