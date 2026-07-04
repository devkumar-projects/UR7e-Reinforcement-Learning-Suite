"""
schema_bloc.py — Schéma bloc de la boucle fermée UR7e laser line-follower.

Génère et sauvegarde :
  - schema_bloc.pdf / .png  (schéma bloc système complet)

Usage :
    python3 -m ur7e_line_follower.schema_bloc
    ros2 run ur7e_line_follower schema_bloc
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

OUT_DIR = Path.home() / '.ros' / 'ur7e_line_follower' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Palette ────────────────────────────────────────────────────────────────────
C = {
    'rl':      '#2E86AB',   # bleu SAC
    'ctrl':    '#A23B72',   # violet régulateur
    'robot':   '#F18F01',   # orange robot
    'ekf':     '#C73E1D',   # rouge EKF
    'cam':     '#3B1F2B',   # noir caméra/KLT
    'sum':     '#444444',   # gris somme
    'ref':     '#1B998B',   # vert référence
    'out':     '#555555',
    'bg':      '#F8F9FA',
    'arrow':   '#333333',
}


def box(ax, x, y, w, h, label, sublabel='', color='#2E86AB', fontsize=9,
        alpha=0.92, radius=0.04):
    fc = color + 'DD'
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f'round,pad={radius}',
                          facecolor=fc, edgecolor=color,
                          linewidth=1.6, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.04 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', zorder=4)
    if sublabel:
        ax.text(x, y - 0.13, sublabel,
                ha='center', va='center', fontsize=6.5,
                color='white', alpha=0.88, zorder=4,
                style='italic')


def circle(ax, x, y, r, label, color='#444444'):
    circ = plt.Circle((x, y), r, facecolor='white',
                       edgecolor=color, linewidth=1.8, zorder=3)
    ax.add_patch(circ)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=11, fontweight='bold', color=color, zorder=4)


def arrow(ax, x0, y0, x1, y1, label='', color='#333333',
          lw=1.4, style='arc3,rad=0.0', zorder=2):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle=style),
                zorder=zorder)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.06, label,
                ha='center', fontsize=6.5, color=color,
                fontstyle='italic', zorder=5)


def dashed_arrow(ax, x0, y0, x1, y1, label='', color='#888888', lw=1.2):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle='dashed',
                                connectionstyle='arc3,rad=0.0'),
                zorder=2)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.06, label,
                ha='center', fontsize=6.5, color=color,
                fontstyle='italic', zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.set_facecolor(C['bg'])
    fig.patch.set_facecolor(C['bg'])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Titre ──────────────────────────────────────────────────────────────────
    ax.text(10, 10.55,
            'Schéma bloc — Boucle fermée UR7e Laser Line-Follower',
            ha='center', va='center', fontsize=13,
            fontweight='bold', color='#222222')
    ax.text(10, 10.2,
            'SAC 2D  ·  MGI différentielle J_wall⁺  ·  EKF cinématique/FK  ·  caméra statique',
            ha='center', va='center', fontsize=8.5, color='#555555')

    # ─────────────────────────────────────────────────────────────────────────
    #  LAYOUT (x, y des centres)
    #  Flux principal de gauche à droite, ligne centrale y=5.5
    # ─────────────────────────────────────────────────────────────────────────

    # 1. Référence trajectoire
    x_ref, y_ref = 1.1, 5.5
    box(ax, x_ref, y_ref, 1.5, 0.8,
        'Trajectoire', 'r(t)  waypoints (y,z)', C['ref'], fontsize=8)

    # 2. Nœud de somme (erreur e = r - ŷ)
    x_sum, y_sum = 3.0, 5.5
    circle(ax, x_sum, y_sum, 0.28, '−', C['sum'])
    ax.text(x_sum - 0.01, y_sum + 0.31, '+', ha='center', fontsize=9,
            fontweight='bold', color=C['sum'])

    # 3. SAC RL Policy
    x_rl, y_rl = 5.1, 5.5
    box(ax, x_rl, y_rl, 1.7, 1.0,
        'Politique SAC', 'π(s) → v_wall∈ℝ²  (acteur MLP 128×128)',
        C['rl'], fontsize=8.5)

    # 4. Régulateur vitesse articulaire
    x_reg, y_reg = 7.6, 5.5
    box(ax, x_reg, y_reg, 1.7, 0.9,
        'MGI + sécurité', 'J_wall⁺·v_wall\nLQR + noyau manipulabilité',
        C['ctrl'], fontsize=8)

    # 5. Robot UR7e (6 axes)
    x_rob, y_rob = 10.3, 5.5
    box(ax, x_rob, y_rob, 2.0, 1.1,
        'Robot  UR7e', 'Simulation Gazebo · Harmonic\n6 joints  q∈ℝ⁶,  q̇∈ℝ⁶',
        C['robot'], fontsize=8.5)

    # 6. FK + projection laser
    x_fk, y_fk = 13.0, 7.2
    box(ax, x_fk, y_fk, 1.9, 0.8,
        'FK + proj. laser', 'T_tcp(q)  →  (y_fk, z_fk)\nσ ≈ 5 mm',
        C['ctrl'], fontsize=7.5)

    # 7. Caméra + KLT tracker
    x_cam, y_cam = 13.0, 3.8
    box(ax, x_cam, y_cam, 1.9, 0.8,
        'Caméra statique', 'HSV laser/ligne + KLT\nobservation relative V2',
        C['cam'], fontsize=7.5)

    # 8. EKF
    x_ekf, y_ekf = 15.8, 5.5
    box(ax, x_ekf, y_ekf, 2.1, 1.8,
        'EKF  (4 états)', 'x = [y, z, vy, vz]\nprédiction J_wall·q̇\ncorrection FK\ncam absolue si calibrée',
        C['ekf'], fontsize=7.5)

    # 9. Sortie — position estimée
    x_out, y_out = 18.7, 5.5
    box(ax, x_out, y_out, 1.5, 0.8,
        'ŷ(t), ẑ(t)', 'Position estimée\n[σ_y, σ_z] incertitude',
        C['ref'], fontsize=7.5)

    # ── Connexions principales (gauche → droite) ───────────────────────────────

    # Référence → somme
    arrow(ax, x_ref + 0.75, y_ref, x_sum - 0.28, y_sum,
          'r(t)', C['ref'])

    # Somme → SAC
    arrow(ax, x_sum + 0.28, y_sum, x_rl - 0.85, y_rl,
          'e(t)∈ℝ²', C['rl'])

    # SAC → Régulateur
    arrow(ax, x_rl + 0.85, y_rl, x_reg - 0.85, y_reg,
          'v_wall∈[−1,1]²', C['rl'])

    # Régulateur → Robot
    arrow(ax, x_reg + 0.85, y_reg, x_rob - 1.0, y_rob,
          'q̇_cmd∈ℝ⁶  (rad/s)', C['ctrl'])

    # Robot → FK (vers haut)
    arrow(ax, x_rob + 0.9, y_rob + 0.2, x_fk - 0.95, y_fk - 0.05,
          'q(t)', C['robot'], style='arc3,rad=-0.2')

    # Robot → Caméra (vers bas)
    arrow(ax, x_rob + 0.9, y_rob - 0.2, x_cam - 0.95, y_cam + 0.05,
          'scène mur + laser', C['robot'], style='arc3,rad=0.2')

    # FK → EKF
    arrow(ax, x_fk + 0.95, y_fk, x_ekf - 1.05, y_ekf + 0.5,
          'z_fk (250 Hz)', C['ctrl'], style='arc3,rad=0.15')

    # Cam → EKF
    arrow(ax, x_cam + 0.95, y_cam, x_ekf - 1.05, y_ekf - 0.5,
          'z_cam (15 Hz)', C['cam'], style='arc3,rad=-0.15')

    # EKF → sortie
    arrow(ax, x_ekf + 1.05, y_ekf, x_out - 0.75, y_out,
          '', C['ekf'])

    # ── Retour d'état vers SAC ─────────────────────────────────────────────────
    # Sortie → bas → retour vers SAC (boucle fermée)
    ax.plot([x_out + 0.75, 19.5, 19.5], [y_out, y_out, 1.2],
            color=C['ekf'], lw=1.5, zorder=2)
    ax.plot([19.5, 5.1], [1.2, 1.2],
            color=C['ekf'], lw=1.5, zorder=2)
    ax.annotate('', xy=(x_rl, y_rl - 0.5),
                xytext=(5.1, 1.2),
                arrowprops=dict(arrowstyle='->', color=C['ekf'], lw=1.5),
                zorder=2)
    ax.text(12.0, 0.85, 'Retour — position estimée  ŷ(t), ẑ(t)  →  état s pour SAC',
            ha='center', fontsize=7.5, color=C['ekf'], fontstyle='italic')

    # ── Retour pour le nœud somme (feedback négatif) ──────────────────────────
    ax.plot([19.5, 19.5], [1.2, 1.2], color=C['ekf'], lw=1.5, zorder=2)
    ax.plot([3.0, 3.0], [1.2, y_sum - 0.28],
            color=C['ref'], lw=1.5, linestyle='--', zorder=2)
    ax.plot([3.0, 19.5], [1.2, 1.2],
            color=C['ref'], lw=1.5, linestyle='--', zorder=2)
    ax.annotate('', xy=(x_sum, y_sum - 0.28),
                xytext=(x_sum, 1.2),
                arrowprops=dict(arrowstyle='->', color=C['ref'], lw=1.5,
                                linestyle='dashed'),
                zorder=2)
    ax.text(x_sum + 0.5, 1.5, 'ŷ(t) feedback', fontsize=6.5,
            color=C['ref'], fontstyle='italic')

    # ── Signal q, q̇ vers EKF (direct depuis robot pour la prédiction) ─────────
    ax.annotate('', xy=(x_ekf - 1.05, y_ekf),
                xytext=(x_rob + 1.0, y_rob),
                arrowprops=dict(arrowstyle='->', color=C['robot'],
                                lw=1.2, linestyle='dashed',
                                connectionstyle='arc3,rad=0.3'),
                zorder=2)
    ax.text(13.3, 6.5, 'q, q̇  (250 Hz)\nprédiction EKF',
            fontsize=6.5, color=C['robot'], fontstyle='italic', ha='center')

    # ── Légende ───────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C['ref'],  label='Référence / sortie estimée'),
        mpatches.Patch(color=C['rl'],   label='Politique SAC (Reinforcement Learning)'),
        mpatches.Patch(color=C['ctrl'], label='Régulateur / cinématique'),
        mpatches.Patch(color=C['robot'],label='Robot UR7e (Gazebo)'),
        mpatches.Patch(color=C['cam'],  label='Caméra + KLT tracker'),
        mpatches.Patch(color=C['ekf'],  label='EKF — filtre de Kalman étendu'),
    ]
    ax.legend(handles=legend_items, loc='lower left',
              bbox_to_anchor=(0.01, 0.01), ncol=2,
              fontsize=7.5, framealpha=0.9, edgecolor='#cccccc')

    # ── Annotations détaillées ────────────────────────────────────────────────
    # État SAC
    ax.text(x_rl, y_rl - 0.7,
            's = [e_y, e_z, q₀..q₅, q̇₀..q̇₅, ŷ, ẑ, vy, vz, σ_y, σ_z, prog]',
            ha='center', fontsize=6.2, color=C['rl'],
            fontstyle='italic')

    # Axes robot
    for i, (name, angle) in enumerate([
        ('J1 — rotation base', '±360°'),
        ('J2 — épaule',        '±360°'),
        ('J3 — coude',         '±360°'),
        ('J4 — poignet 1',     '±360°'),
        ('J5 — poignet 2',     '±360°'),
        ('J6 — poignet 3',     '±360°'),
    ]):
        ax.text(10.3, 4.55 - i * 0.15, f'  {name}  {angle}',
                ha='center', fontsize=5.5, color='#555555')

    # Bloc EKF détaillé
    ax.text(x_ekf, y_ekf - 1.15,
            'Prédiction : ẋ = F·x + J_wall(q)·q̇\n'
            'Mesure FK  : z_fk  ~  N(y_fk, 5mm²)\n'
            'Mesure cam : z_cam ~  N(y_cam,15mm²)\n'
            'Gain Kalman K optimal  (forme Joseph)',
            ha='center', fontsize=5.8, color=C['ekf'],
            fontstyle='italic',
            bbox=dict(fc='white', ec=C['ekf'], alpha=0.6, boxstyle='round,pad=0.2'))

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        p = OUT_DIR / f'schema_bloc.{ext}'
        fig.savefig(p, dpi=180, bbox_inches='tight')
        print(f'[schema_bloc] Sauvegardé → {p}')

    plt.close(fig)


def entry_point():
    main()


if __name__ == '__main__':
    main()
