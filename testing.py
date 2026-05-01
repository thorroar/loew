import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import loew

np.random.seed(123)
t = np.linspace(0.0, 1.0, 200)  # keep low for speed, increase later
fBm = loew.misc.brownian_motion(t, kappa=3)

n_frames = 40  # number of animation frames
frame_indices = np.linspace(2, len(t) - 1, n_frames).astype(int)

# ── Grid setup ────────────────────────────────────────────────────────────────
numbergridx = 30
numbergridy = 30
x = np.concatenate([np.linspace(-3, -0.5, numbergridx), np.linspace(-0.5, 0.5, numbergridx), np.linspace(0.5, 3, numbergridx)])
y = np.linspace(0.05, 3, numbergridy)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# ── Precompute grid for every frame ──────────────────────────────────────────
print("Precomputing grids...")
grids = []
for idx in frame_indices:
    t_frame = t[:idx]
    u_frame = fBm[:idx]
    Z_mapped = Z.copy()
    for i in range(len(t_frame) - 1, 0, -1):
        dt = t_frame[i] - t_frame[i - 1]
        du = u_frame[i] - u_frame[i - 1]
        Z_mapped = loew.chordal.vslit_zip(Z_mapped, dt, du)
    grids.append(Z_mapped)
print("Done.")

# ── Precompute full trace for axis limits ─────────────────────────────────────
asle = loew.chordal.trace(t, fBm)

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')
ax.set_aspect('equal')
ax.set_title('Chordal SLE — Conformal Map Growing', color='white', fontsize=13)
ax.set_xlabel('$Re$', color='white', fontsize='x-large')
ax.set_ylabel('$Im$', color='white', fontsize='x-large')

pad = 0.3
ax.set_xlim(asle.real.min() - pad, asle.real.max() + pad)
ax.set_ylim(-0.1, asle.imag.max() + pad)

def update(frame):
    ax.cla()
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_aspect('equal')
    ax.set_title('Chordal SLE — Conformal Map Growing', color='white', fontsize=13)
    ax.set_xlabel('$Re$', color='white', fontsize='x-large')
    ax.set_ylabel('$Im$', color='white', fontsize='x-large')
    ax.set_xlim(asle.real.min() - pad, asle.real.max() + pad)
    ax.set_ylim(-0.1, asle.imag.max() + pad)

    Z_mapped = grids[frame]
    idx = frame_indices[frame]

    # Draw grid
    for i, row in enumerate(Z_mapped):
        color = plt.cm.cool(i / len(Z_mapped))
        ax.plot(row.real, row.imag, color=color, lw=0.5, alpha=0.6)
    for j, col in enumerate(Z_mapped.T):
        color = plt.cm.cool(j / Z_mapped.shape[1])
        ax.plot(col.real, col.imag, color=color, lw=0.5, alpha=0.6)

    # Draw trace up to this frame
    pts = np.array([asle.real[:idx], asle.imag[:idx]]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap='autumn', linewidth=2, zorder=5)
    lc.set_array(t[:idx-1])
    ax.add_collection(lc)

ani = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
ani.save('sle_growing_grid.gif', writer=PillowWriter(fps=12))
print("Saved sle_growing_grid.gif")
plt.show()