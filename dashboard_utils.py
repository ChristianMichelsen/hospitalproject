import extra_funcs
import pandas as pd
from scipy import special
import shap
import numpy as np
from collections import namedtuple
import plotly.graph_objects as go
import plotly.express as px


#%%


def get_trained_model(d_in):

    X_train = d_in["X_train_val"]
    y_train = d_in["y_train_val"]

    X_test = d_in["X_test"]
    y_test = d_in["y_test"]

    y_pred_proba = d_in["y_pred_proba"]
    d_optuna_all = d_in["d_optuna_all"]

    d_data = d_in["d_data"]
    df_shap = d_in["df_shaps"]

    cutoff = extra_funcs.compute_cutoff(y_test, y_pred_proba)

    model = extra_funcs.FLModel(d_optuna_all, d_data).fit()

    return model, cutoff, X_train.columns


#%%


def add_bmi_to_dict(d):
    d["bmi"] = d["weight"] / (d["height"] / 100) ** 2


def add_date_info_to_dict(d):
    d["year"] = d["date"].year
    d["month"] = d["date"].month
    d["day_of_week"] = d["date"].weekday()
    d["day_of_month"] = d["date"].day
    d["year_fraction"] = d["date"].timetuple().tm_yday / 366 + d["year"]
    del d["date"]


def dict_to_patient(d_patient, columns):

    add_bmi_to_dict(d_patient)
    add_date_info_to_dict(d_patient)

    X_patient = pd.DataFrame.from_dict(d_patient, orient="index").T
    X_patient = X_patient[columns]

    return X_patient


#%%


def get_shap_patient(model, cutoff, X_patient):

    init_score = model.fl.init_score(model.d_data["y_train_val"])
    offset = init_score - special.logit(cutoff)

    explainer = shap.TreeExplainer(model.model)
    shap_patient = explainer(X_patient)
    shap_patient.base_values += offset

    return shap_patient[0]


#%%


ShapCollection = namedtuple(
    "ShapCollection",
    ("shaps " "shaps_cumsum " "names " "data_hover_names " "N_display_all"),
)


def get_shap_plot_object(shap_values, max_display=10):

    base_value = shap_values.base_values
    data_values = shap_values.data
    names = np.array(shap_values.feature_names)
    shaps = shap_values.values

    N_display = min(max_display, len(shaps))
    N_rest = len(shaps) - N_display
    N_display_all = N_display + 2

    order_abs = np.argsort(-np.abs(shaps))

    shap_rest = shaps.sum() - shaps[order_abs][:N_display].sum()

    data_values = data_values[order_abs][:N_display]
    names = names[order_abs][:N_display]
    shaps = shaps[order_abs][:N_display]

    order_min2max = np.argsort(shaps)
    names = names[order_min2max]
    data_values = data_values[order_min2max]
    shaps = shaps[order_min2max]

    names = ["base value", f"{N_rest} other variables", *names]
    data_values = np.array([None, None, *data_values])
    shaps = np.array([base_value, shap_rest, *shaps])

    shaps_cumsum = np.array([0, *np.cumsum(shaps)[:-1]])

    data_hover_names = []
    for name, value in zip(names, data_values):
        if value is not None:
            if value.is_integer():
                value = int(value)
            else:
                value = round(value, 2)

            s = f"{name} = {value}"
        else:
            s = ""
        data_hover_names.append(s)

    return ShapCollection(
        shaps,
        shaps_cumsum,
        names,
        data_hover_names,
        N_display_all,
    )


#%%


def plot_shap_object(shap_collection):

    shaps = shap_collection.shaps
    shaps_cumsum = shap_collection.shaps_cumsum
    names = shap_collection.names
    data_hover_names = shap_collection.data_hover_names
    N_display_all = shap_collection.N_display_all

    colors = px.colors.qualitative.Set1
    blue = colors[1]
    red = colors[0]

    marker_colors = [blue if val < 0 else red for val in shaps]

    fig = go.Figure(
        go.Bar(
            x=shaps,
            y=names,
            base=shaps_cumsum,
            text=shaps,
            textposition="auto",
            orientation="h",
            marker_color=marker_colors,
            showlegend=False,
            customdata=np.stack((shaps, data_hover_names)).T,
            hovertemplate=(
                "<b>%{y}</b><br><br>"
                "%{customdata[1]}<br>"
                "Shap = %{customdata[0]:.3f} <extra></extra>"
            ),
        ),
    )

    final_shap = (shaps_cumsum + shaps)[-1]
    vline_color = red if final_shap > 0 else blue
    fig.add_vline(
        final_shap,
        line_width=3,
        line_dash="dash",
        line_color=vline_color,
        annotation_text="Final Prediction",
        annotation_position="left",
        annotation_font_color=vline_color,
    )

    xmin = np.min(shaps_cumsum)
    xmax = np.max(shaps_cumsum + shaps)

    xlim = [xmin - abs(xmin) * 0.15, xmax + abs(xmax) * 0.15]

    dtick = 0.05
    tick0 = np.ceil(xlim[0] / dtick) * dtick
    tick1 = np.ceil(xlim[1] / dtick) * dtick
    tickvals = np.arange(tick0, tick1, dtick)

    fig.add_trace(
        go.Scatter(
            x=tickvals,
            y=[N_display_all] * len(tickvals),
            showlegend=False,
            opacity=0,
            hoverinfo="skip",
            xaxis="x2",
        ),
    )

    # Add shape regions
    fig.add_vrect(
        x0=xlim[0],
        x1=0,
        fillcolor=blue,
        opacity=0.05,
        layer="below",
        line_width=0,
    )

    # Add shape regions
    fig.add_vrect(
        x0=0,
        x1=xlim[1],
        fillcolor=red,
        opacity=0.05,
        layer="below",
        line_width=0,
    )

    tickvals2_text = [f"{tickval*100:.2f}%" for tickval in special.expit(tickvals)]

    fig.update_layout(
        title="Prediction explanation",
        title_x=0.05,
        title_y=0.95,
        title_font_size=20,
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        xaxis={
            "title": "logit",
            "range": xlim,
            "tickvals": tickvals,
        },
        yaxis_range=(-0.5, 11.7),
        xaxis2={
            "title": "Probability",
            "showgrid": False,
            "anchor": "y",
            "overlaying": "x",
            "side": "top",
            "tickvals": tickvals,
            "ticktext": tickvals2_text,
            "range": xlim,
        },
        hovermode="closest",
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
    )

    fig.update_yaxes(automargin=True)

    return fig