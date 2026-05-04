import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import loew

# ── Flags ─────────────────────────────────────────────────────────────────────
show_grid = False
n_frames = 100  # number of animation frames

np.random.seed(123)
t = np.linspace(0.0, 1.0, 1000)  # keep reasonable for speed
fBm = loew.misc.brownian_motion(t, kappa=5)

# ── Precompute full trace ─────────────────────────────────────────────────────
asle = loew.chordal.trace(t, fBm)

# ── Precompute grid (static — doesn't change) ─────────────────────────────────
if show_grid:
    numbergridx = 20
    numbergridy = 20
    x = np.concatenate([np.linspace(-3, -0.5, numbergridx), np.linspace(-0.5, 0.5, numbergridx), np.linspace(0.5, 3, numbergridx)])
    y = np.linspace(0.05, 3, numbergridy)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    Z_mapped = Z.copy()
    for i in range(len(t) - 1, 0, -1):
        dt = t[i] - t[i - 1]
        du = fBm[i] - fBm[i - 1]
        Z_mapped = loew.chordal.vslit_zip(Z_mapped, dt, du)

# ── Set up figure ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')
ax.set_aspect('equal')
ax.set_title('Chordal SLE — Growing Trace', color='white', fontsize=13)
ax.set_xlabel('$Re$', color='white', fontsize='x-large')
ax.set_ylabel('$Im$', color='white', fontsize='x-large')

# Fix axis limits to full trace so it doesn't jump around
pad = 0.2
ax.set_xlim(asle.real.min() - pad, asle.real.max() + pad)
ax.set_ylim(asle.imag.min() - pad, asle.imag.max() + pad)

# Draw static grid
if show_grid:
    for i, row in enumerate(Z_mapped):
        color = plt.cm.plasma(i / len(Z_mapped))
        ax.plot(row.real, row.imag, color=color, lw=0.6, alpha=0.4)
    for j, col in enumerate(Z_mapped.T):
        color = plt.cm.cool(j / Z_mapped.shape[1])
        ax.plot(col.real, col.imag, color=color, lw=0.6, alpha=0.4)

# Trace line that will grow
trace_line, = ax.plot([], [], color='orange', lw=1.5, zorder=5)

# ── Animation ─────────────────────────────────────────────────────────────────
frame_indices = np.linspace(1, len(t) - 1, n_frames).astype(int)

def update(frame):
    idx = frame_indices[frame]
    # color trace by time using LineCollection
    ax.collections = [c for c in ax.collections if not hasattr(c, '_is_trace')]
    pts = np.array([asle.real[:idx], asle.imag[:idx]]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap='autumn', linewidth=2, zorder=5)
    lc.set_array(t[:idx-1])
    lc._is_trace = True
    ax.add_collection(lc)
    return lc,

ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)

# ── Save ──────────────────────────────────────────────────────────────────────
ani.save('sle_growth.gif', writer=PillowWriter(fps=20))
print("Saved sle_growth.gif")
plt.show()