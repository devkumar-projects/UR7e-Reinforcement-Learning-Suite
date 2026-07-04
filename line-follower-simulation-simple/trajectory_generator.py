"""
Générateur de trajectoires aléatoires 2D — UR7e
─────────────────────────────────────────────────
Produit des lignes / courbes dans le plan du "tableau"
(z constant), garanties dans l'espace atteignable du robot.

Types de trajectoires :
  - segment droit
  - courbe de Bézier (1 ou 2 points de contrôle)
  - sinusoïde
  - arc de cercle

Sortie : array (N, 3) de points 3D le long de la trajectoire.

Testable seul :
    python3 trajectory_generator.py
"""
import numpy as np

# ─────────────────────────────────────────────
# Paramètres du "tableau" (plan vertical devant le robot)
# Le tableau est un plan à X fixe, on dessine dans (Y, Z)
# ─────────────────────────────────────────────
TABLEAU_X     = 0.45          # distance du tableau devant la base (m)
TABLEAU_Y_MIN = -0.35         # bord gauche
TABLEAU_Y_MAX =  0.35         # bord droit
TABLEAU_Z_MIN =  0.25         # bas
TABLEAU_Z_MAX =  0.75         # haut

# Espace atteignable (cohérent avec ur7e_env_v4)
R_MIN = 0.20
R_MAX = 0.85

# Discrétisation par défaut
DEFAULT_N_POINTS = 60
POINT_SPACING    = 0.025      # ~2.5 cm entre points (indicatif)


# ─────────────────────────────────────────────
# Vérification d'atteignabilité
# ─────────────────────────────────────────────
def _is_reachable(point):
    """Vérifie qu'un point est dans la sphère atteignable."""
    r = np.linalg.norm(point)
    return R_MIN <= r <= R_MAX

def _clip_to_tableau(y, z):
    """Contraint un point dans les bornes du tableau."""
    y = np.clip(y, TABLEAU_Y_MIN, TABLEAU_Y_MAX)
    z = np.clip(z, TABLEAU_Z_MIN, TABLEAU_Z_MAX)
    return y, z


# ─────────────────────────────────────────────
# Générateurs de formes (dans le plan Y-Z)
# ─────────────────────────────────────────────
def _segment(rng, n):
    """Ligne droite entre deux points aléatoires du tableau."""
    y0 = rng.uniform(TABLEAU_Y_MIN, TABLEAU_Y_MAX)
    z0 = rng.uniform(TABLEAU_Z_MIN, TABLEAU_Z_MAX)
    y1 = rng.uniform(TABLEAU_Y_MIN, TABLEAU_Y_MAX)
    z1 = rng.uniform(TABLEAU_Z_MIN, TABLEAU_Z_MAX)
    t  = np.linspace(0, 1, n)
    ys = y0 + t * (y1 - y0)
    zs = z0 + t * (z1 - z0)
    return ys, zs

def _bezier(rng, n):
    """Courbe de Bézier quadratique (3 points de contrôle)."""
    p0 = np.array([rng.uniform(TABLEAU_Y_MIN, TABLEAU_Y_MAX),
                   rng.uniform(TABLEAU_Z_MIN, TABLEAU_Z_MAX)])
    p1 = np.array([rng.uniform(TABLEAU_Y_MIN, TABLEAU_Y_MAX),
                   rng.uniform(TABLEAU_Z_MIN, TABLEAU_Z_MAX)])
    p2 = np.array([rng.uniform(TABLEAU_Y_MIN, TABLEAU_Y_MAX),
                   rng.uniform(TABLEAU_Z_MIN, TABLEAU_Z_MAX)])
    t  = np.linspace(0, 1, n).reshape(-1, 1)
    pts = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
    return pts[:, 0], pts[:, 1]

def _bezier_cubic(rng, n):
    """Courbe de Bézier cubique (4 points de contrôle) — plus sinueuse."""
    pts_ctrl = [
        np.array([rng.uniform(TABLEAU_Y_MIN, TABLEAU_Y_MAX),
                  rng.uniform(TABLEAU_Z_MIN, TABLEAU_Z_MAX)])
        for _ in range(4)
    ]
    p0, p1, p2, p3 = pts_ctrl
    t  = np.linspace(0, 1, n).reshape(-1, 1)
    pts = ((1-t)**3 * p0 + 3*(1-t)**2*t * p1
           + 3*(1-t)*t**2 * p2 + t**3 * p3)
    return pts[:, 0], pts[:, 1]

def _sinusoide(rng, n):
    """Sinusoïde horizontale dans le tableau."""
    y0 = TABLEAU_Y_MIN
    y1 = TABLEAU_Y_MAX
    ys = np.linspace(y0, y1, n)
    z_center = rng.uniform(TABLEAU_Z_MIN + 0.1, TABLEAU_Z_MAX - 0.1)
    amp      = rng.uniform(0.05, 0.15)
    freq     = rng.uniform(1.0, 3.0)
    phase    = rng.uniform(0, 2*np.pi)
    zs = z_center + amp * np.sin(freq * 2*np.pi * (ys - y0)/(y1 - y0) + phase)
    return ys, zs

def _arc(rng, n):
    """Arc de cercle dans le tableau."""
    cy = rng.uniform(TABLEAU_Y_MIN + 0.1, TABLEAU_Y_MAX - 0.1)
    cz = rng.uniform(TABLEAU_Z_MIN + 0.1, TABLEAU_Z_MAX - 0.1)
    radius = rng.uniform(0.08, 0.18)
    a0 = rng.uniform(0, 2*np.pi)
    a1 = a0 + rng.uniform(np.pi/2, 3*np.pi/2)
    angles = np.linspace(a0, a1, n)
    ys = cy + radius * np.cos(angles)
    zs = cz + radius * np.sin(angles)
    return ys, zs


SHAPE_GENERATORS = {
    'segment'  : _segment,
    'bezier'   : _bezier,
    'bezier3'  : _bezier_cubic,
    'sinus'    : _sinusoide,
    'arc'      : _arc,
}


# ─────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────
def generate_trajectory(
    shape=None,
    n_points=DEFAULT_N_POINTS,
    seed=None,
    x_plane=TABLEAU_X,
):
    """
    Génère une trajectoire 2D dans le plan du tableau (X fixe).

    Args:
        shape     : type de courbe ('segment','bezier','bezier3','sinus','arc')
                    ou None pour un choix aléatoire
        n_points  : nombre de points de discrétisation
        seed      : graine aléatoire (reproductibilité)
        x_plane   : position X du tableau (profondeur)

    Returns:
        trajectory : array (n_points, 3) de points 3D [x, y, z]
        shape_name : nom de la forme générée
    """
    rng = np.random.default_rng(seed)

    if shape is None:
        shape = rng.choice(list(SHAPE_GENERATORS.keys()))

    generator = SHAPE_GENERATORS[shape]

    # Générer la forme 2D (dans le plan Y-Z)
    ys, zs = generator(rng, n_points)

    # Contraindre dans les bornes du tableau
    ys = np.clip(ys, TABLEAU_Y_MIN, TABLEAU_Y_MAX)
    zs = np.clip(zs, TABLEAU_Z_MIN, TABLEAU_Z_MAX)

    # Construire les points 3D (X = profondeur du tableau)
    xs = np.full(n_points, x_plane)
    trajectory = np.column_stack([xs, ys, zs]).astype(np.float32)

    # Vérifier l'atteignabilité de tous les points
    reachable = np.array([_is_reachable(p) for p in trajectory])
    if not reachable.all():
        # Si certains points sont hors de portée, réduire l'amplitude
        # en rapprochant du centre du tableau
        center = np.array([x_plane,
                           (TABLEAU_Y_MIN + TABLEAU_Y_MAX)/2,
                           (TABLEAU_Z_MIN + TABLEAU_Z_MAX)/2])
        for i in range(n_points):
            if not _is_reachable(trajectory[i]):
                # Rapprocher du centre jusqu'à atteignabilité
                for alpha in np.linspace(0.1, 1.0, 10):
                    candidate = (1-alpha)*trajectory[i] + alpha*center
                    if _is_reachable(candidate):
                        trajectory[i] = candidate
                        break

    return trajectory, shape


def trajectory_length(trajectory):
    """Longueur totale de la trajectoire (m)."""
    diffs = np.diff(trajectory, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))


# ─────────────────────────────────────────────
# Test autonome
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("Test du générateur de trajectoires\n")

    for shape in SHAPE_GENERATORS.keys():
        traj, name = generate_trajectory(shape=shape, seed=42)
        length = trajectory_length(traj)
        print(f"  {name:10s} : {len(traj)} points | "
              f"longueur {length*100:.1f} cm | "
              f"X={traj[0,0]:.2f} | "
              f"Y∈[{traj[:,1].min():.2f},{traj[:,1].max():.2f}] | "
              f"Z∈[{traj[:,2].min():.2f},{traj[:,2].max():.2f}]")

    # Test trajectoire aléatoire
    print("\n5 trajectoires aléatoires :")
    for i in range(5):
        traj, name = generate_trajectory(seed=i)
        print(f"  {i}: {name:10s} | {trajectory_length(traj)*100:.1f} cm")

    # Visualisation optionnelle
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(SHAPE_GENERATORS),
                                  figsize=(20, 4))
        for ax, shape in zip(axes, SHAPE_GENERATORS.keys()):
            traj, name = generate_trajectory(shape=shape, seed=7)
            ax.plot(traj[:, 1], traj[:, 2], 'b-o', markersize=2)
            ax.scatter(traj[0, 1], traj[0, 2], c='green', s=80,
                       zorder=5, label='début')
            ax.scatter(traj[-1, 1], traj[-1, 2], c='red', s=80,
                       zorder=5, label='fin')
            ax.set_title(name)
            ax.set_xlabel('Y (m)')
            ax.set_ylabel('Z (m)')
            ax.set_xlim(TABLEAU_Y_MIN-0.05, TABLEAU_Y_MAX+0.05)
            ax.set_ylim(TABLEAU_Z_MIN-0.05, TABLEAU_Z_MAX+0.05)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig('trajectoires_exemples.png', dpi=120, bbox_inches='tight')
        print("\nVisualisation sauvegardée : trajectoires_exemples.png")
    except ImportError:
        print("\n(matplotlib absent — pas de visualisation)")

    print("\nTest terminé !")
