# ==============================================================================
# FICHIER : fred_potential_fields.py
# RÔLE : Champs de potentiel (attractifs + répulsifs) pour l'adaptation à l'UR7e sous PyBullet.
#
# FIDÈLE à robot_env_utils.py de Fred :
#   - forces attractives : tirent les points de contrôle vers leurs cibles,
#     saturées au-delà d'un cutoff (get_attractive_force_world).
#   - forces répulsives : via p.getClosestPoints, formule de Khatib bornée
#     (1/d - 1/d0)*(1/d^2), clippée (get_repulsive_forces_world).
#   - TOUT EN CENTIMÈTRES (comme Fred : positions * 100).
#
# Différence d'adaptation : Fred a son robot maison ; ici on cible les links
# de l'UR7e (effecteur, poignet, coude) identifiés par leurs noms d'URDF.
# ==============================================================================

import numpy as np
import pybullet as p

# Conversion mètres -> centimètres (Fred travaille en cm)
M_TO_CM = 100.0


def get_control_point_pos(robot_body_id, link_index, physics_client_id):
    """Position 3D (en CM) d'un point de contrôle (link) du robot."""
    ls = p.getLinkState(robot_body_id, link_index,
                        physicsClientId=physics_client_id)
    return np.array(ls[0]) * M_TO_CM


def get_attractive_force_world(control_points, target_points,
                               attractive_cutoff_distance=10.0, weights=None):
    """Vecteurs attractifs des points de contrôle vers leurs cibles.
    Norme bornée à attractive_cutoff_distance. FIDÈLE à Fred.

    Args:
        control_points : (N,3) positions actuelles des points de contrôle (cm).
        target_points  : (N,3) positions cibles des points de contrôle (cm).
        attractive_cutoff_distance : distance au-delà de laquelle la norme sature.
        weights        : (N,) importance relative de chaque point.
    Returns:
        forces (N,3), total_distance (somme des distances aux cibles).
    """
    n = control_points.shape[0]
    if weights is None:
        weights = np.ones(n)
    forces = np.zeros((n, 3))
    total_distance = 0.0
    for i in range(n):
        vector = control_points[i] - target_points[i]
        distance = np.linalg.norm(vector)
        if distance == 0:
            pass
        elif distance > attractive_cutoff_distance:
            forces[i] = -attractive_cutoff_distance * weights[i] * vector / distance
        else:
            forces[i] = -weights[i] * vector
        total_distance += distance
    return forces, total_distance


def _normal_and_distance(robot_body_id, obstacle_id, link_index,
                         physics_client_id, repulsive_cutoff_distance):
    """Normale et distance (en CM) entre un link et un obstacle, via PyBullet.
    FIDÈLE à Fred (get_normal_and_distance)."""
    res = p.getClosestPoints(bodyA=robot_body_id, bodyB=obstacle_id,
                             linkIndexA=link_index,
                             distance=repulsive_cutoff_distance / M_TO_CM,
                             physicsClientId=physics_client_id)
    if res == ():
        return [0, 0, 1], 1e7
    # normal_on_b à l'index 7, distance à l'index 8
    _, _, _, _, _, _, _, normal_on_b, d, *rest = res[0]
    return normal_on_b, d * M_TO_CM


def get_repulsive_forces_world(robot_body_id, link_indices, link_radii,
                               link_weights, obstacle_ids, physics_client_id,
                               repulsive_cutoff_distance=8.0, clip_force=6.0):
    """Vecteurs répulsifs (Khatib borné) pour chaque link face aux obstacles.
    FIDÈLE à Fred (get_repulsive_forces_world).

    Pour chaque link : on cherche l'obstacle le plus proche, et si la distance
    (surface) est sous le cutoff, on applique force = w*(1/d - 1/d0)*(1/d^2),
    clippée, dirigée selon la normale obstacle->link.
    """
    n = len(link_indices)
    forces = np.zeros((n, 3))
    for i in range(n):
        link_index = link_indices[i]
        smallest_distance = repulsive_cutoff_distance
        closest_normal = None
        for obstacle_id in obstacle_ids:
            normal, d = _normal_and_distance(robot_body_id, obstacle_id,
                                             link_index, physics_client_id,
                                             repulsive_cutoff_distance)
            distance = d - link_radii[i]
            if distance < 0:           # le link chevauche l'obstacle
                distance = 0.1
            if distance < smallest_distance:
                closest_normal = normal
                smallest_distance = distance
        if smallest_distance < repulsive_cutoff_distance and closest_normal is not None:
            d = smallest_distance
            term = link_weights[i] * (1.0 / d - 1.0 / repulsive_cutoff_distance) \
                * (1.0 / (d * d))
            term = np.clip(term, 0, clip_force)
            forces[i] = term * np.array(closest_normal)
    return forces


def normalize_vector_for_obs(vec):
    """Normalise un vecteur seulement s'il dépasse 1 (FIDÈLE à Fred :
    _get_normalized_vector_as_list). Garde les petites valeurs telles quelles."""
    norm = np.linalg.norm(vec)
    if norm < 1.0:
        return vec.tolist()
    return (vec / norm).tolist()
