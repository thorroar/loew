import numpy as np
import matplotlib.pyplot as plt
import loew

np.random.seed(1233)
kappa = 2
t = np.linspace(0.0, 1.0, 1000)
fBm = loew.misc.brownian_motion(t, kappa)

# ── Precompute trace ──────────────────────────────────────────────────────────
print("Computing trace...")
asle = loew.chordal.trace(t, fBm)
print("Done.")

# ── Dense pixel grid ──────────────────────────────────────────────────────────
res = 1000
x = np.linspace(-2, 2, res)
y = np.linspace(0.001, 3, res)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# ── Apply forward Loewner map g_t by mirroring the drive loop ─────────────────
print("Computing forward map...")
z = asle.copy()
Z_mapped = Z.copy()
alive = np.ones((res, res), dtype=bool)

for i, w in enumerate(z[1:], start=1):
    x_tip = w.real
    y_tip = w.imag
    z[i+1:] = loew.chordal.vslit_unzip(z[i+1:], x_tip, y_tip)
    Z_mapped = loew.chordal.vslit_unzip(Z_mapped, x_tip, y_tip)
    alive &= (Z_mapped.imag > -0.01)

print("Done.")

# ── Color by log(Im(g_t(z))) cyclically ───────────────────────────────────────
im_part = np.where(alive, Z_mapped.imag, np.nan)
log_im = np.log(np.abs(im_part))
period = 0.5
hue = (log_im % period) / period

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 10), facecolor='black')
ax.set_facecolor('black')

ax.imshow(
    hue,
    cmap='hsv',
    origin='lower',
    extent=[x.min(), x.max(), y.min(), y.max()],
    aspect='auto',
    interpolation='bilinear',
    vmin=0, vmax=1
)

ax.set_title(f'SLE $\\kappa={kappa}$ — domain coloring of $\\log(\\mathrm{{Im}}(g_t(z))))$',
             color='white', fontsize=12)
ax.set_xlabel('$Re$', color='white', fontsize='x-large')
ax.set_ylabel('$Im$', color='white', fontsize='x-large')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')

fig.tight_layout()
plt.savefig('sle_domain_coloring.png', dpi=200, bbox_inches='tight',
            facecolor='black')
plt.show()