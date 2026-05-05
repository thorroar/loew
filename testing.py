import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import loew
import sys
sys.path.append('/Users/thor/Desktop/Coding/Python/Numerical analysis/ContOpt_HW2/loew')

def radial_vslit_unzip(z, w_tip):
    """
    Inverse of radial vslit_zip — maps D\slit back to D.
    w_tip is the slit tip (a complex number inside the unit disk).
    Uses the Möbius transformation that maps w_tip to 0, applies
    the square root, then maps back.
    """
    # Möbius transform sending w_tip -> 0
    phi = (z - w_tip) / (1 - np.conj(w_tip) * z)
    # square root branch that keeps points in D
    w = np.sqrt(phi)
    # correct branch: pick the root closer to z
    mask = np.abs(w + np.sqrt(-phi)) < np.abs(w - np.sqrt(-phi))
    w[mask] = -w[mask]
    # inverse Möbius
    return (w + w_tip) / (1 + np.conj(w_tip) * w)

# ── Setup ─────────────────────────────────────────────────────────────────────
seed_value = 1
kappa = 1
np.random.seed(seed_value)
t = np.linspace(0.0, 1.0, 100)
U = loew.misc.brownian_motion(t, kappa)

print("Computing radial trace...")
asle = loew.radial.trace(t, U)
print("Done.")

# ── Pixel grid — unit disk ────────────────────────────────────────────────────
res = 500
x = np.linspace(-1.0, 1.0, res)
y = np.linspace(-1.0, 1.0, res)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# mask outside unit disk
outside = np.abs(Z) >= 1.0

# ── Precompute frames incrementally ──────────────────────────────────────────
n_frames = 50
frame_steps = np.linspace(1, len(t) - 1, n_frames).astype(int)

print("Precomputing frames...")
z = asle.copy()
Z_mapped = Z.copy()
# alive: inside disk and not swallowed by hull
alive = ~outside.copy()

frames_hue = []
prev_step = 1

for frame_idx, step in enumerate(frame_steps):
    for i in range(prev_step, step):
        w_tip = z[i]  # current slit tip in already-unzipped coordinates
        # unzip remaining trace points
        z[i+1:] = radial_vslit_unzip(z[i+1:], w_tip)
        # apply same unzip to pixel grid
        Z_mapped = radial_vslit_unzip(Z_mapped, w_tip)
        # pixels swallowed by hull leave the unit disk
        alive &= (np.abs(Z_mapped) < 1.0 + 0.01)
    prev_step = step

    # color by log(1 - |g_t(z)|) — distance to boundary of disk
    dist = 1.0 - np.abs(Z_mapped)
    dist = np.where(alive & ~outside, dist, np.nan)
    log_dist = np.log(np.abs(dist))
    period = 0.5
    hue = (log_dist % period) / period
    frames_hue.append(hue.copy())

    if frame_idx % 10 == 0:
        print(f"Frame {frame_idx}/{n_frames} done.")

print("All frames precomputed.")

# ── Animate ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
ax.set_facecolor('black')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')
ax.set_xlabel('$Re$', color='white', fontsize='x-large')
ax.set_ylabel('$Im$', color='white', fontsize='x-large')

# draw unit circle
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), color='white', lw=0.5, alpha=0.3)

img = ax.imshow(
    frames_hue[0],
    cmap='hsv',
    origin='lower',
    extent=[x.min(), x.max(), y.min(), y.max()],
    aspect='equal',
    interpolation='bilinear',
    vmin=0, vmax=1
)

ax.set_title(f'Radial SLE $\\kappa={kappa}$ — domain coloring of $\\log(1-|g_t(z)|)$',
             color='white', fontsize=12)

time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                    color='white', fontsize=11, va='top',
                    fontfamily='monospace')

def update(frame):
    img.set_data(frames_hue[frame])
    t_val = t[frame_steps[frame]]
    time_text.set_text(f't = {t_val:.3f}')
    return img, time_text

ani = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)

print("Saving animation...")
ani.save(f'radial_sle_animation_{res}_{n_frames}_{seed_value}_{kappa}_.gif',
         writer=PillowWriter(fps=10))
print(f"Saved radial_sle_animation_{res}_{n_frames}_{seed_value}_{kappa}_.gif")
plt.show()