import pandas as pd
import shap
import sklearn
import numpy as np
import copy

#%%

# a classic housing price dataset
X, y = shap.datasets.boston()
X = X.loc[["CRIM", "LSTAT", "TAX", "B"]]
X100 = shap.utils.sample(X, 100)  # 100 instances for use as the background distribution

# a simple linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)


# # fit a GAM model to the data
# import interpret.glassbox
# model_ebm = interpret.glassbox.ExplainableBoostingRegressor()
# model_ebm.fit(X, y)

# explain the GAM model with SHAP
# explainer_ebm = shap.Explainer(model_ebm.predict, X100)
# shap_values_ebm = explainer_ebm(X)

# the waterfall_plot shows how we get from explainer.expected_value to model.predict(X)[sample_ind]
# shap.plots.beeswarm(shap_values, max_display=14)


X = X.iloc[[0, 2, 6, 8, 10]]
X.loc[6, "TAX"] = 100


#%%

# compute the SHAP values for the linear model
explainer = shap.Explainer(model.predict, X100)
shap_values = explainer(X)


np.random.random(())

values = np.array([])



shap_values.values[:, 0] += np.arange(4)
shap_values.values[:, -1] += np.arange(4)[::1] / 2


#%%

for i in range(1, len(X.columns) + 1):
    shap.plots.beeswarm(copy.deepcopy(shap_values), max_display=i)
# %%


x=x


shap.plots.bar(shap_values)
shap.plots.bar(shap_values.abs.max(0))

shap_values = explainer(X)
shap.plots.beeswarm(shap_values, max_display=14)
# shap_values = explainer(X)
# shap.plots.beeswarm(shap_values.abs.max(0), max_display=14)

#%%'

np.set_printoptions(suppress=True)

from shap import Explanation
import matplotlib.pyplot as pl

import scipy as sp
from shap.plots._utils import convert_ordering, sort_inds, get_sort_order, convert_color

from shap.plots._labels import labels

from shap.utils import safe_isinstance
from shap.plots import colors

shap_values = explainer(X)
shap_values.values[:, 0] += np.arange(4)
shap_values.values[:, -1] += np.arange(4)[::1] / 2

max_display = 3
order = Explanation.abs.mean(0)
alpha = 1
color_bar = True
color_bar_label = labels["FEATURE_VALUE"]
axis_color = "#333333"

# support passing an explanation object
shap_exp = shap_values
values = shap_exp.values
features = shap_exp.data
feature_names = shap_exp.feature_names

order = convert_ordering(order, values)

# default color:
color = colors.red_blue
color = convert_color(color)

# determine how many top features we will plot
num_features = min(max_display, len(feature_names))

# iteratively merge nodes until we can cut off the smallest feature values to stay within
# num_features without breaking a cluster tree
orig_inds = [[i] for i in range(len(feature_names))]
orig_values = values.copy()
feature_order = convert_ordering(order, Explanation(np.abs(values)))

# here we build our feature names, accounting for the fact that some features might be merged together
feature_inds = feature_order[:max_display]
y_pos = np.arange(len(feature_inds), 0, -1)
feature_names_new = []
for pos, inds in enumerate(orig_inds):
    if len(inds) == 1:
        feature_names_new.append(feature_names[inds[0]])
    elif len(inds) <= 2:
        feature_names_new.append(" + ".join([feature_names[i] for i in inds]))
    else:
        max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
        feature_names_new.append(
            feature_names[inds[max_ind]] + " + %d other features" % (len(inds) - 1)
        )
feature_names = feature_names_new

# see how many individual (vs. grouped at the end) features we are plotting
if num_features < len(values[0]):
    num_cut = np.sum(
        [
            len(orig_inds[feature_order[i]])
            for i in range(num_features - 1, len(values[0]))
        ]
    )
    values[:, feature_order[num_features - 1]] = np.sum(
        [values[:, feature_order[i]] for i in range(num_features - 1, len(values[0]))],
        0,
    )

row_height = 0.4
pl.gcf().set_size_inches(8, min(len(feature_order), max_display) * row_height + 1.5)
pl.axvline(x=0, color="#999999", zorder=-1)

# make the beeswarm dots
for pos, i in enumerate(reversed(feature_inds)):

    # break

    pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    shaps = values[:, i]
    fvalues = features[:, i]
    inds = np.arange(len(shaps))
    np.random.shuffle(inds)
    fvalues = fvalues[inds]
    shaps = shaps[inds]

    N = len(shaps)
    nbins = 100
    quant = np.round(
        nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8)
    )
    inds = np.argsort(quant + np.random.randn(N) * 1e-6)
    layer = 0
    last_bin = -1
    ys = np.zeros(N)
    for ind in inds:
        if quant[ind] != last_bin:
            layer = 0
        ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[ind]
    ys *= 0.9 * (row_height / np.max(ys + 1))

    # trim the color range, but prevent the color range from collapsing
    vmin = np.nanpercentile(fvalues, 5)
    vmax = np.nanpercentile(fvalues, 95)
    if vmin == vmax:
        vmin = np.nanpercentile(fvalues, 1)
        vmax = np.nanpercentile(fvalues, 99)
        if vmin == vmax:
            vmin = np.min(fvalues)
            vmax = np.max(fvalues)
    if vmin > vmax:  # fixes rare numerical precision issues
        vmin = vmax

    assert features.shape[0] == len(
        shaps
    ), "Feature and SHAP matrices must have the same number of rows!"

    # plot the nan fvalues in the interaction feature as grey
    nan_mask = np.isnan(fvalues)
    pl.scatter(
        shaps[nan_mask],
        pos + ys[nan_mask],
        color="#777777",
        vmin=vmin,
        vmax=vmax,
        s=16,
        alpha=alpha,
        linewidth=0,
        zorder=3,
        rasterized=len(shaps) > 500,
    )

    # plot the non-nan fvalues colored by the trimmed feature value
    cvals = fvalues[np.invert(nan_mask)].astype(np.float64)
    cvals_imp = cvals.copy()
    cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
    cvals[cvals_imp > vmax] = vmax
    cvals[cvals_imp < vmin] = vmin
    pl.scatter(
        shaps[np.invert(nan_mask)],
        pos + ys[np.invert(nan_mask)],
        cmap=color,
        vmin=vmin,
        vmax=vmax,
        s=16,
        c=cvals,
        alpha=alpha,
        linewidth=0,
        zorder=3,
        rasterized=len(shaps) > 500,
    )


# draw the color bar
import matplotlib.cm as cm

m = cm.ScalarMappable(cmap=color)
m.set_array([0, 1])
cb = pl.colorbar(m, ticks=[0, 1], aspect=1000)
cb.set_ticklabels([labels["FEATURE_VALUE_LOW"], labels["FEATURE_VALUE_HIGH"]])
cb.set_label(color_bar_label, size=12, labelpad=0)
cb.ax.tick_params(labelsize=11, length=0)
cb.set_alpha(1)
cb.outline.set_visible(False)
bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
cb.ax.set_aspect((bbox.height - 0.9) * 20)
# cb.draw_all()

pl.gca().xaxis.set_ticks_position("bottom")
pl.gca().yaxis.set_ticks_position("none")
pl.gca().spines["right"].set_visible(False)
pl.gca().spines["top"].set_visible(False)
pl.gca().spines["left"].set_visible(False)
pl.gca().tick_params(color=axis_color, labelcolor=axis_color)


# build our y-tick labels
yticklabels = [feature_names[i] for i in feature_inds]
if num_features < len(values[0]):
    yticklabels[-1] = "Sum of %d other features" % num_cut
pl.yticks(range(len(feature_inds)), reversed(yticklabels), fontsize=13)

pl.gca().tick_params("y", length=20, width=0.5, which="major")
pl.gca().tick_params("x", labelsize=11)
pl.ylim(-1, len(feature_inds))
pl.xlabel(labels["VALUE"], fontsize=13)

pl.show()

# %%
