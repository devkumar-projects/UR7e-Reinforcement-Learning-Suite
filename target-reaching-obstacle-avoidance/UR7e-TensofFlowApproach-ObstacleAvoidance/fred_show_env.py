# ==============================================================================
# FICHIER : fred_show_env.py
# RÔLE : Affiche STATIQUEMENT l'environnement dans PyBullet — cobot en position
#        de repos + obstacle (selon la phase) + marqueurs départ/arrivée.
#        AUCUN mouvement, aucune politique. Juste pour inspecter la géométrie.
#
# Usage : python fred_show_env.py 2     (phase 1, 2 ou 3 ; défaut 2)
#   - phase 1 : pas d'obstacle, cobot au repos seul
#   - phase 2 : obstacle + départ (vert) et arrivée (bleu) fixes
#   - phase 3 : obstacle + départ fixe (vert) + zone cible gauche (points bleus)
# ==============================================================================

import os, sys, time
import numpy as np
import pybullet as p
import pybullet_data

PHASE = sys.argv[1] if len(sys.argv) > 1 else "2"

WORKSPACE = {"x": (-0.42, 0.42), "y": (0.45, 0.82), "z": (0.15, 0.58)}
FLOOR_Z_CM = WORKSPACE["z"][0] * 100.0
HOME = [0.0, -1.0, 1.0, -1.57, -1.57, 0.0]
CONTROLLABLE_JOINTS = [2, 3, 4, 5, 6, 7]

# obstacle phases 2-3
BOX_DIMS_CM = (5.0, 27.0, 35.0)
BOX_CENTER_XY_CM = (0.0, 60.5)
RIGHT_X_CM = (BOX_DIMS_CM[0] / 2 + WORKSPACE["x"][1] * 100) / 2
LEFT_X_CM = -RIGHT_X_CM
MID_Y_CM = (WORKSPACE["y"][0] + WORKSPACE["y"][1]) * 100 / 2


def add_sphere(pos_cm, rgba, radius=0.025):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    p.createMultiBody(0, -1, vis, basePosition=[c / 100.0 for c in pos_cm])


def draw_workspace_box():
    """Trace les arêtes du volume de travail (lignes de debug)."""
    x0, x1 = WORKSPACE["x"]; y0, y1 = WORKSPACE["y"]; z0, z1 = WORKSPACE["z"]
    corners = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
               (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        p.addUserDebugLine(corners[a], corners[b], [0.5,0.5,0.5], 1.0)


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.loadURDF("plane.urdf")

    urdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ur7e_pybullet")
    cwd = os.getcwd()
    os.chdir(urdf_dir)
    robot = p.loadURDF("ur7e.urdf", [0, 0, 0],
                       p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
    os.chdir(cwd)

    # cobot en position de repos
    for k, j in enumerate(CONTROLLABLE_JOINTS):
        p.resetJointState(robot, j, HOME[k], 0.0)

    draw_workspace_box()

    # obstacle (phases 2-3)
    if PHASE in ("2", "3"):
        half = [d / 200.0 for d in BOX_DIMS_CM]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                  rgbaColor=[0.9, 0.2, 0.2, 0.85])
        center_cm = [BOX_CENTER_XY_CM[0], BOX_CENTER_XY_CM[1], BOX_DIMS_CM[2]/2.0]
        p.createMultiBody(0, col, vis,
                          basePosition=[c/100.0 for c in center_cm])

    # marqueurs départ/arrivée
    if PHASE == "2":
        #add_sphere([RIGHT_X_CM, MID_Y_CM, FLOOR_Z_CM], [0,1,0,1])   # départ vert
        #add_sphere([LEFT_X_CM, MID_Y_CM, FLOOR_Z_CM], [0,0,1,1])    # arrivée bleu

        START_X_CM = RIGHT_X_CM + 10.0
        TARGET_X_CM = LEFT_X_CM - 10.0
        add_sphere([START_X_CM, MID_Y_CM, FLOOR_Z_CM], [0,1,0,1])   # départ vert
        add_sphere([TARGET_X_CM, MID_Y_CM, FLOOR_Z_CM], [0,0,1,1])    # arrivée bleu
    elif PHASE == "3":
        add_sphere([RIGHT_X_CM, MID_Y_CM, FLOOR_Z_CM], [0,1,0,1])   # départ vert
        # zone cible gauche : nuage de points bleus possibles
        x_inner = -(BOX_DIMS_CM[0]/2 + 2.0); x_outer = WORKSPACE["x"][0]*100
        for xc in np.linspace(x_outer, x_inner, 4):
            for yc in np.linspace(WORKSPACE["y"][0]*100, WORKSPACE["y"][1]*100, 4):
                add_sphere([xc, yc, FLOOR_Z_CM], [0,0,1,0.6], radius=0.015)

    # caméra
    p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=50,
                                 cameraPitch=-30, cameraTargetPosition=[0,0.38,0.3])

    print(f"=== Environnement PHASE {PHASE} ===")
    print("Cobot en position de repos.", end=" ")
    if PHASE == "1":
        print("Pas d'obstacle (cibles aléatoires dans le volume).")
    elif PHASE == "2":
        print("Obstacle + départ (vert) et arrivée (bleu) fixes.")
    else:
        print("Obstacle + départ fixe (vert) + zone cible gauche (points bleus).")
    print("Volume de travail tracé en gris. Ferme la fenêtre pour quitter.")

    while p.isConnected():
        p.stepSimulation()
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
