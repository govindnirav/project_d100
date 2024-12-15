# %%
from project_d100.data import load_parquet
from project_d100.evaluation import evaluate_predictions
from project_d100.modelling import load_model

# %%
# Load the best pipelines
glm_best_pipeline = load_model("glm_best_pipeline")
lgbm_best_pipeline = load_model("lgbm_best_pipeline")

df_all = load_parquet("hours_cleaned.parquet")

# %%
# Evaluate the GLM model
evaluate_predictions(
    df_all, "cnt", glm_best_pipeline, train_size=0.8, stratify="cnt", n_bins=50
)

# %%
# Evaluate the LGBM model
evaluate_predictions(
    df_all, "cnt", lgbm_best_pipeline, train_size=0.8, stratify="cnt", n_bins=50
)
