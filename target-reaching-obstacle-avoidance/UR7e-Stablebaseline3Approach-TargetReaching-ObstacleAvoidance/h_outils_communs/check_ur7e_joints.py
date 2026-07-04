# ==============================================================================
# FICHIER : check_ur7e_joints.py
# RÔLE : Vérifier les indices des joints contrôlables et de l'effecteur du UR7e.
#        On NE SUPPOSE PAS qu'ils sont identiques au UR30 — on vérifie.
# ==============================================================================

import os
import pybullet as p
import pybullet_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_DIR = os.path.join(SCRIPT_DIR, "ur7e_pybullet")

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
cwd = os.getcwd()
os.chdir(URDF_DIR)
robot = p.loadURDF("ur7e.urdf", [0, 0, 0], useFixedBase=True)
os.chdir(cwd)

type_names = {p.JOINT_REVOLUTE: "REVOLUTE", p.JOINT_PRISMATIC: "PRISMATIC",
              p.JOINT_FIXED: "FIXED"}
controllable = []
ee_index = None

print(f"{'idx':>3} | {'joint':<28} | {'type':<9} | enfant")
print("-" * 70)
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    name = info[1].decode()
    jtype = info[2]
    child = info[12].decode()
    print(f"{i:>3} | {name:<28} | {type_names.get(jtype,'?'):<9} | {child}")
    if jtype == p.JOINT_REVOLUTE:
        controllable.append(i)
    if child == "tool0":
        ee_index = i

print("-" * 70)
print(f">>> Joints contrôlables (REVOLUTE) : {controllable}")
print(f">>> Index de tool0 (effecteur)     : {ee_index}")
print(f">>> Attendu (comme UR30) : [2,3,4,5,6,7] et tool0=10")
if controllable == [2, 3, 4, 5, 6, 7] and ee_index == 10:
    print(">>> IDENTIQUE au UR30 : aucune modification d'indices nécessaire.")
else:
    print(">>> DIFFÉRENT du UR30 : il faudra ajuster les indices dans bullet_ur7e.py")
p.disconnect()
