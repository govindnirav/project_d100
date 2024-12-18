from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from project_d100.data import log_transform
from project_d100.modelling import glm_pipeline, lgbm_pipeline, load_model, load_split
from project_d100.visualisations import (
    plot_box,
    plot_count,
    plot_cramerv,
    plot_histogram,
    save_graph,
)

CLEAN = Path(__file__).parent.parent / "data" / "cleaned"
GRAPH_PATH = Path(__file__).parent.parent / "visualisations"

# Load
df_unclean = pl.read_parquet(CLEAN / "hours_partially_cleaned.parquet")
df_clean = pl.read_parquet(CLEAN / "hours_cleaned.parquet")

X_train, X_test, y_train, y_test = load_split()

predictors = [col for col in X_train.columns if col != "instant"]
categoricals = [
    "season",
    "yr",
    "mnth",
    "hr",
    "weekday",
    "workingday",
    "weathersit",
]
numericals = [col for col in predictors if col not in categoricals]

glm_pipeline = glm_pipeline(numericals, categoricals)
lgbm_pipeline = lgbm_pipeline(numericals, categoricals)

glm_best_pipeline = load_model("glm_best_pipeline")
lgbm_best_pipeline = load_model("lgbm_best_pipeline")

# Unclean Data Plots

# Target Variable
# Histogram and boxplot of cnt and log_cnt

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
plot_histogram(df_unclean, "cnt", bins=100, ax=axes[0], fontsize=20)
plot_box(
    df_unclean,
    x_var="cnt",
    title="Boxplot of log transformed cnt (whiskers at 1.5 IQR)",
    ax=axes[1],
    fontsize=20,
)
plt.tight_layout()
save_graph(GRAPH_PATH, plt, "hist_box_cnt")

df_unclean = log_transform(df_unclean, "cnt")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
plot_histogram(df_unclean, "log_cnt", bins=100, ax=axes[0], fontsize=20)
plot_box(
    df_unclean,
    x_var="log_cnt",
    title="Boxplot of log transformed cnt (whiskers at 1.5 IQR)",
    ax=axes[1],
    fontsize=20,
)
plt.tight_layout()
save_graph(GRAPH_PATH, plt, "hist_box_log_cnt")

# Feature Variables
# Humidity
plot_histogram(
    df_unclean, "hum", bins=30, title="Humidity Distribution", xlabel="Humidity (%)"
)
save_graph(GRAPH_PATH, plt, "hist_hum")

# Holiday
plot_count(df_unclean, "holiday")
save_graph(GRAPH_PATH, plt, "count_holiday")

# Day
plot_box(
    df_unclean,
    x_var="day",
    y_var="cnt",
    whis=(0, 100),
    orient="v",
    title="Boxplot of cnt by day (whiskers at 0 and 100 percentiles)",
)
save_graph(GRAPH_PATH, plt, "box_cnt_day")

# Clean Data Plots
# Hour
plot_box(
    df_clean,
    x_var="cnt",
    y_var="hr",
    whis=(0, 100),
    title="Boxplot of cnt by hour (whiskers at 0 and 100 percentiles)",
)
save_graph(GRAPH_PATH, plt, "box_cnt_hr")

# Cramer's V
plot_cramerv(df_clean, index_var="instant")
save_graph(GRAPH_PATH, plt, "cramerv")

# Pipeline Plots
# %%
glm_pipeline
lgbm_pipeline

# Evaluation Plots
# %%
