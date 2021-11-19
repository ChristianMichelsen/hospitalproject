import numpy as np
import matplotlib.pyplot as plt

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

fig, ax = plt.subplots(figsize=(6, 0.5))
ax.set_title("Patient", fontsize=14)
ax.imshow(gradient, aspect="auto", cmap="coolwarm_r")
ax.set_axis_off()
ax.axvline(256 / 2, ls="--", c="k")
ax.scatter(50, 0.5, c="k", s=50)

fig.savefig("figures/example.pdf", bbox_inches="tight")
