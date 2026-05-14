"""
Geometric EL Embeddings: Training Animation
===========================================

All three geometric EL models embed ontology classes as 2-D shapes when
trained with ``embed_dim=2``, letting us watch the geometry evolve directly —
no dimensionality reduction needed.

* **ELEm** — classes as *circles* (centre ``class_embed``, radius ``class_rad``)
* **ELBE** — classes as *axis-aligned rectangles* (centre ``class_embed``, half-extents ``class_offset``)
* **Box²EL** — classes as *axis-aligned rectangles* (centre ``class_center``, half-extents ``class_offset``)

This example trains all three models on the Family ontology and saves an
animated GIF showing how the shapes evolve across epochs.
"""

# %%
# Imports and JVM initialisation
# ------------------------------

import mowl
mowl.init_jvm("10g")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import torch as th

from mowl.datasets.builtin import FamilyDataset
from mowl.models import ELEmbeddings, ELBE, BoxSquaredEL
from mowl.visualization.el.base import _local_name

# %%
# Dataset and entity selection
# ----------------------------
# We use a small fixed set of classes so the animation stays readable.

dataset = FamilyDataset()

# %%
# Training parameters
# -------------------

EPOCHS = 100
SNAPSHOT_EVERY = 5   # capture a frame every N epochs
N_CLASSES = 6        # number of classes to visualise

# %%
# Snapshot helpers
# ----------------
# Instead of storing patches (which need an active axes), we store raw numpy
# arrays and recreate patches at animation time.

def _indices(model, n):
    """Return the first *n* class indices and their IRIs."""
    items = list(model.class_index_dict.items())[:n]
    iris = [iri for iri, _ in items]
    idxs = th.tensor([idx for _, idx in items])
    return iris, idxs


def _elem_snapshot(model, idxs):
    with th.no_grad():
        centers = model.module.class_embed.weight[idxs].cpu().numpy()   # (N, 2)
        radii   = np.abs(model.module.class_rad.weight[idxs].cpu().numpy())  # (N, 1)
    return centers, radii


def _box_snapshot_elbe(model, idxs):
    with th.no_grad():
        centers = model.module.class_embed.weight[idxs].cpu().numpy()
        halves  = np.abs(model.module.class_offset.weight[idxs].cpu().numpy())
    return centers, halves


def _box_snapshot_box2(model, idxs):
    with th.no_grad():
        centers = model.module.class_center.weight[idxs].cpu().numpy()
        halves  = np.abs(model.module.class_offset.weight[idxs].cpu().numpy())
    return centers, halves


def make_callback(snapshots, getter, idxs):
    def callback(epoch, model):
        if epoch % SNAPSHOT_EVERY == 0:
            snapshots.append(getter(model, idxs))
    return callback


# %%
# Train ELEm
# ----------

elem_model = ELEmbeddings(dataset, embed_dim=2)
iris, idxs = _indices(elem_model, N_CLASSES)
labels = [_local_name(iri) for iri in iris]

elem_snapshots = []
elem_model.train(
    epochs=EPOCHS,
    epoch_callback=make_callback(elem_snapshots, _elem_snapshot, idxs),
)

# %%
# Train ELBE
# ----------

elbe_model = ELBE(dataset, embed_dim=2)
elbe_snapshots = []
elbe_model.train(
    epochs=EPOCHS,
    epoch_callback=make_callback(elbe_snapshots, _box_snapshot_elbe, idxs),
)

# %%
# Train Box²EL
# ------------

box2_model = BoxSquaredEL(dataset, embed_dim=2)
box2_snapshots = []
box2_model.train(
    epochs=EPOCHS,
    epoch_callback=make_callback(box2_snapshots, _box_snapshot_box2, idxs),
)

# %%
# Build the animation
# -------------------

colors = plt.cm.tab10(np.linspace(0, 0.6, N_CLASSES))

def _axis_limits(all_snaps, pad=0.3):
    """Compute a common axis limit across all snapshots and models."""
    xs, ys = [], []
    for snaps in all_snaps:
        for centers, sizes in snaps:
            if sizes.shape[1] == 1:          # ELEm: radius scalar
                r = sizes[:, 0]
                xs.extend((centers[:, 0] - r).tolist())
                xs.extend((centers[:, 0] + r).tolist())
                ys.extend((centers[:, 1] - r).tolist())
                ys.extend((centers[:, 1] + r).tolist())
            else:                            # ELBE / Box²EL: half-extents
                xs.extend((centers[:, 0] - sizes[:, 0]).tolist())
                xs.extend((centers[:, 0] + sizes[:, 0]).tolist())
                ys.extend((centers[:, 1] - sizes[:, 1]).tolist())
                ys.extend((centers[:, 1] + sizes[:, 1]).tolist())
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    return xmin, xmax, ymin, ymax


xmin, xmax, ymin, ymax = _axis_limits(
    [elem_snapshots, elbe_snapshots, box2_snapshots]
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
model_titles = ["ELEm (circles)", "ELBE (boxes)", "Box²EL (boxes)"]
all_snaps = [elem_snapshots, elbe_snapshots, box2_snapshots]


def draw_frame(frame_idx):
    for ax, snaps, title in zip(axes, all_snaps, model_titles):
        ax.cla()
        ax.set_title(title, fontsize=11)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        centers, sizes = snaps[frame_idx]
        for i, (cx, cy) in enumerate(centers):
            color = colors[i]
            if sizes.shape[1] == 1:          # circle
                r = float(sizes[i, 0])
                patch = mpatches.Circle((cx, cy), r, fill=False,
                                        edgecolor=color, linewidth=1.5)
            else:                            # rectangle
                hx, hy = float(sizes[i, 0]), float(sizes[i, 1])
                patch = mpatches.Rectangle(
                    (cx - hx, cy - hy), 2 * hx, 2 * hy,
                    fill=False, edgecolor=color, linewidth=1.5,
                )
            ax.add_patch(patch)
            ax.annotate(labels[i], (cx, cy), ha="center", va="center",
                        fontsize=7, color=color)

    epoch = frame_idx * SNAPSHOT_EVERY
    fig.suptitle(f"Epoch {epoch}", fontsize=13)


n_frames = len(elem_snapshots)
ani = animation.FuncAnimation(fig, draw_frame, frames=n_frames, interval=150)

# %%
# Save and display
# ----------------

gif_path = "el_geometry_training.gif"
ani.save(gif_path, writer=animation.PillowWriter(fps=8))
print(f"Saved animation to {gif_path}")

# Show the final frame as a static image for the gallery thumbnail
draw_frame(n_frames - 1)
fig.tight_layout()
plt.show()
