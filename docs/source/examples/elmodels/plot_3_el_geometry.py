"""
Geometric EL Embeddings: Training Animation
===========================================

All three geometric EL models embed ontology classes as 2-D shapes when
trained with ``embed_dim=2``, letting us watch the geometry evolve directly —
no dimensionality reduction needed.

* **ELEm** — classes as *circles* (centre ``class_embed``, radius ``class_rad``)
* **ELBE** — classes as *axis-aligned rectangles* (centre ``class_embed``, half-extents ``class_offset``)
* **Box²EL** — classes as *axis-aligned rectangles* (centre ``class_center``, half-extents ``class_offset``)

This example trains all three models on the Family ontology and produces an
interactive animation showing how the shapes evolve across epochs.
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
from IPython.display import HTML

from mowl.datasets.builtin import FamilyDataset
from mowl.models import ELEmbeddings, ELBE, BoxSquaredEL
from mowl.visualization.el.base import _local_name

# %%
# Dataset and entity selection
# ----------------------------

dataset = FamilyDataset()

# %%
# Training parameters
# -------------------

EPOCHS = 100
SNAPSHOT_EVERY = 5
N_CLASSES = 6

# %%
# Snapshot helpers
# ----------------
# We store raw numpy arrays per snapshot and recreate patches at animation time.

def _indices(model, n):
    items = list(model.class_index_dict.items())[:n]
    iris = [iri for iri, _ in items]
    idxs = th.tensor([idx for _, idx in items])
    return iris, idxs


def _elem_snapshot(model, idxs):
    with th.no_grad():
        centers = model.module.class_embed.weight[idxs].cpu().numpy()
        radii   = np.abs(model.module.class_rad.weight[idxs].cpu().numpy())
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
elem_model.train(epochs=EPOCHS,
                 epoch_callback=make_callback(elem_snapshots, _elem_snapshot, idxs))

# %%
# Train ELBE
# ----------

elbe_model = ELBE(dataset, embed_dim=2)
elbe_snapshots = []
elbe_model.train(epochs=EPOCHS,
                 epoch_callback=make_callback(elbe_snapshots, _box_snapshot_elbe, idxs))

# %%
# Train Box²EL
# ------------

box2_model = BoxSquaredEL(dataset, embed_dim=2)
box2_snapshots = []
box2_model.train(epochs=EPOCHS,
                 epoch_callback=make_callback(box2_snapshots, _box_snapshot_box2, idxs))

# %%
# Build and display the animation
# --------------------------------

colors = plt.cm.tab10(np.linspace(0, 0.6, N_CLASSES))


def _axis_limits(all_snaps, pad=0.3):
    xs, ys = [], []
    for snaps in all_snaps:
        for centers, sizes in snaps:
            if sizes.shape[1] == 1:
                r = sizes[:, 0]
                xs.extend((centers[:, 0] - r).tolist())
                xs.extend((centers[:, 0] + r).tolist())
                ys.extend((centers[:, 1] - r).tolist())
                ys.extend((centers[:, 1] + r).tolist())
            else:
                xs.extend((centers[:, 0] - sizes[:, 0]).tolist())
                xs.extend((centers[:, 0] + sizes[:, 0]).tolist())
                ys.extend((centers[:, 1] - sizes[:, 1]).tolist())
                ys.extend((centers[:, 1] + sizes[:, 1]).tolist())
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


xmin, xmax, ymin, ymax = _axis_limits([elem_snapshots, elbe_snapshots, box2_snapshots])

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
            if sizes.shape[1] == 1:
                r = float(sizes[i, 0])
                patch = mpatches.Circle((cx, cy), r, fill=False,
                                        edgecolor=color, linewidth=1.5)
            else:
                hx, hy = float(sizes[i, 0]), float(sizes[i, 1])
                patch = mpatches.Rectangle(
                    (cx - hx, cy - hy), 2 * hx, 2 * hy,
                    fill=False, edgecolor=color, linewidth=1.5)
            ax.add_patch(patch)
            ax.annotate(labels[i], (cx, cy), ha="center", va="center",
                        fontsize=7, color=color)
    fig.suptitle(f"Epoch {frame_idx * SNAPSHOT_EVERY}", fontsize=13)


n_frames = len(elem_snapshots)
ani = animation.FuncAnimation(fig, draw_frame, frames=n_frames, interval=150)

# %%
# Sphinx-Gallery picks up the IPython HTML output and embeds it as an
# interactive animation with play/pause controls.

HTML(ani.to_jshtml())
