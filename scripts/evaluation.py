# %%
import warnings
from pathlib import Path

import dalex as dx
import matplotlib.pyplot as plt
import plotly.io as pio

from project_d100.data import load_parquet
from project_d100.evaluation import evaluate_predictions
from project_d100.modelling import load_model, load_split
from project_d100.visualisations import (
    plot_lorenz_curve,
    plot_partial_dependence,
    plot_preds_actual,
    plot_shapley,
    plot_variable_importance,
    save_graph,
)

# %%
# Mute warnings - neater output
warnings.filterwarnings("ignore")

# Graph path
GRAPH_PATH = Path(__file__).parent.parent / "visualisations"

# %%
# Load the best pipelines
glm_best_pipeline = load_model("glm_best_pipeline")
lgbm_best_pipeline = load_model("lgbm_best_pipeline")

df_all = load_parquet("hours_cleaned.parquet")

# Load the split data
X_train, X_test, y_train, y_test = load_split()

# %%
# Evaluate the best GLM and LGBM models
glm_eval, glm_test, glm_preds = evaluate_predictions(
    df_all, "cnt", glm_best_pipeline, X_test, y_test
)

lgbm_eval, lgbm_test, lgbm_preds = evaluate_predictions(
    df_all, "cnt", lgbm_best_pipeline, X_test, y_test
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
save_graph(GRAPH_PATH, plt, "lorenz_curve")
plt.show()

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
save_graph(GRAPH_PATH, plt, "preds_actual")
plt.show()

# %%
# GLM and LGBM Dalex Explainers
X_test_f = X_test.astype(float)
glm_explainer = dx.Explainer(glm_best_pipeline, X_test_f, y_test, label="GLM")
lgbm_explainer = dx.Explainer(lgbm_best_pipeline, X_test_f, y_test, label="LGBM")
# %%
# Feature relevance for GLM and LGBM
fig1 = plot_variable_importance(explainer=glm_explainer, model_name="GLM")
pio.write_image(fig=fig1, file=GRAPH_PATH / "vi_glm.png", format="png", scale=2)
fig1.show()

fig2 = plot_variable_importance(explainer=lgbm_explainer, model_name="LGBM")
pio.write_image(fig=fig2, file=GRAPH_PATH / "vi_lgbm.png", format="png", scale=2)
fig2.show()
# Top 5 features for LGBM: *** (not the same as GLM)
# hr, workingday, temp, yr, weathersit

# %%
# Partial dependence plots for GLM and LGBM (for top 5 features (LGBM))
fig3 = plot_partial_dependence(
    glmexplainer=glm_explainer,
    lgbmexplainer=lgbm_explainer,
    features=["hr", "workingday", "temp", "yr", "weathersit"],
    type="pdp",
)
pio.write_image(fig=fig3, file=GRAPH_PATH / "pdp.png", format="png", scale=2)
fig3.show()
# Many of the pdps are useless because the features are categorical

fig4 = plot_partial_dependence(
    glmexplainer=glm_explainer,
    lgbmexplainer=lgbm_explainer,
    features=["hr", "workingday", "temp", "yr", "weathersit"],
    type="pdp",
)
pio.write_image(fig=fig4, file=GRAPH_PATH / "ale.png", format="png", scale=2)
fig4.show()

# %%
# Shapeley values for GLM and LGBM
plot_shapley(
    pipeline=glm_best_pipeline, X_test=X_test, max_display=10, model_name="GLM"
)
plt.savefig(GRAPH_PATH / "glm_shapely.png", format="png", dpi=300, bbox_inches="tight")
plt.show()

plot_shapley(
    pipeline=lgbm_best_pipeline, X_test=X_test, max_display=10, model_name="LGBM"
)
plt.savefig(GRAPH_PATH / "lgbm_shapely.png", format="png", dpi=300, bbox_inches="tight")
plt.show()
# %%
