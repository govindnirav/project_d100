# %%
from project_d100.data import load_parquet
from project_d100.evaluation import evaluate_predictions
from project_d100.modelling import load_model
from project_d100.visualisations import plot_lorenz_curve, plot_preds_actual

# %%
# Load the best pipelines
glm_best_pipeline = load_model("glm_best_pipeline")
lgbm_best_pipeline = load_model("lgbm_best_pipeline")

df_all = load_parquet("hours_cleaned.parquet")

# %%
# Evaluate the best GLM and LGBM models
glm_eval, glm_test, glm_preds = evaluate_predictions(
    df_all, "cnt", glm_best_pipeline, train_size=0.8, stratify="cnt", n_bins=50
)
lgbm_eval, lgbm_test, lgbm_preds = evaluate_predictions(
    df_all, "cnt", lgbm_best_pipeline, train_size=0.8, stratify="cnt", n_bins=50
)

# %%
# Compare lorenz curves for GLM and LGBM
plot_lorenz_curve(
    glm_test,  # Same as lgbm because seeded
    glm_preds,
    "GLM",
    lgbm_test,
    lgbm_preds,
    "LGBM",
)

# %%
# Compare predictions vs actuals plot for GLM and LGBM
plot_preds_actual(
    glm_test,
    glm_preds,
    "GLM",
    lgbm_test,
    lgbm_preds,
    "LGBM",
    axis=(0, 1000),
)
