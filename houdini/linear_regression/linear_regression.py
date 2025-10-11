import json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path

CSV = Path("C:/Users/kko8/OneDrive/projects/houdini_snippets/prod/3d/scenes/ML/Lab_1/data/triarea.csv")


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


def scatter_with_fitted_line():
    """
    What it shows: linear relation and any obvious outliers.
    """

    x = df["X_volume"].to_numpy(); y = df["y_mass"].to_numpy()

    rho = float(np.dot(x,y) / np.dot(x,x))  # slope through origin
    xx = np.linspace(x.min(), x.max(), 100)

    plt.figure(); plt.scatter(x, y); plt.plot(xx, rho*xx)
    plt.xlabel("Volume (m³)"); plt.ylabel("Mass (kg)"); plt.title("Mass vs Volume")
    plt.tight_layout(); plt.show()


def Histogram_of_log10():
    """
    What it shows: if most values are “near zero” or just right-skewed. A log histogram spreads them out.
    """

    x = df["X_volume"].to_numpy()
    plt.figure()
    plt.hist(np.log10(x[x > 0]), bins=30)
    plt.xlabel("log10(Volume m³)")
    plt.ylabel("Count")
    plt.title("Distribution on log scale")
    plt.tight_layout()
    plt.show()


def plot_scatter2(x, y, x_label, y_label, title,
                 hue=None, size=None, logx=False, logy=False, bins=20):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- scatter ---
    if hue is None:
        axes[0].scatter(x, y)
    else:
        # normalize hue to [0,1] and color by it; add a colorbar
        h = np.asarray(hue)
        sc = axes[0].scatter(x, y, c=h)
        cb = fig.colorbar(sc, ax=axes[0])
        cb.set_label("hue")

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


# plot_scatter(df["X_volume"], df["y_mass"], "Volume (X_volume)", "Mass (y_mass)", "Mass vs Volume")
print_df_info(df)
# scatter_with_fitted_line()
# Histogram_of_log10()
# plot_scatter2(df["X_area"], df["y_total_prims"], "X_area", "y_total_prims", "Total Primitives vs Area",hue=df["iteration"])
