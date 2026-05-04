import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import loew


seed_value =1000
kappa = 6
np.random.seed(seed_value)
t = np.linspace(0.0, 1, 1000)
fBm = loew.misc.brownian_motion(t, kappa)

print("Computing trace...")
asle = loew.chordal.trace(t, fBm)
print("Done.")

res = 2000
x = np.linspace(-2, 2, res)
y = np.linspace(0.001, 3, res)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# ── Precompute all frames incrementally ──────────────────────────────────────
n_frames = 50
frame_steps = np.linspace(1, len(t) - 1, n_frames).astype(int)

print("Precomputing frames...")
z = asle.copy()
Z_mapped = Z.copy()
alive = np.ones((res, res), dtype=bool)

frames_hue = []
prev_step = 1

for frame_idx, step in enumerate(frame_steps):
    for i in range(prev_step, step):
        w = z[i]
        x_tip = w.real
        y_tip = w.imag
        z[i+1:] = loew.chordal.vslit_unzip(z[i+1:], x_tip, y_tip)
        Z_mapped = loew.chordal.vslit_unzip(Z_mapped, x_tip, y_tip)
        alive &= (Z_mapped.imag > -0.01)
    prev_step = step

    im_part = np.where(alive, Z_mapped.imag, np.nan)
    log_im = np.log(np.abs(im_part))
    period = 0.5
    hue = (log_im % period) / period
    frames_hue.append(hue.copy())

    if frame_idx % 10 == 0:
        print(f"Frame {frame_idx}/{n_frames} done.")

print("All frames precomputed.")

# ── Animate ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 10), facecolor='black')
ax.set_facecolor('black')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')
ax.set_xlabel('$Re$', color='white', fontsize='x-large')
ax.set_ylabel('$Im$', color='white', fontsize='x-large')

img = ax.imshow(
    frames_hue[0],
    cmap='hsv',
    origin='lower',
    extent=[x.min(), x.max(), y.min(), y.max()],
    aspect='auto',
    interpolation='bilinear',
    vmin=0, vmax=1
)

ax.set_title(f'SLE $\\kappa={kappa}$ — domain coloring of $\\log(\\mathrm{{Im}}(g_t(z)))$',
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
ani.save(f'chordal_sle_animation{res}_{n_frames}_{seed_value}_{kappa}_.gif', writer=PillowWriter(fps=10))
print(f"Saved chordal_sle_animation{res}_{n_frames}_{seed_value}_{kappa}_.gif")
plt.show()