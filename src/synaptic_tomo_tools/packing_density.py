import numpy as np

def calculate_packing_density_using_sliding_cylinder(
    active_zone: dict,
    active_zonogram: dict,
    aunp_coordinates: np.ndarray,
    cylinder_radius: float = 25.0,
    receptor_crosssection_nm_squared: float = 122.0,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree
    from synaptic_tomo_tools.activezone import transform_coordinates_to_active_zonogram
    
    ps_mesh = active_zone['active_postsynaptic_mesh']
    subset_vertices = ps_mesh.vertices[::50]  # Use only x and y coordinates for 2D projection

    
    tree = cKDTree(ps_mesh.vertices)
    #Generate a cKDTree of aunps
    tree_aunps = cKDTree(aunp_coordinates)
    # Iterate of vertices in ps_mesh_simplified and find all vertices in ps_mesh within 25 and average their normals
    num_aunps_at_vertex = []
    for v in subset_vertices:
        idxs = tree.query_ball_point(v, cylinder_radius)
        normals = ps_mesh.vertex_normals[idxs]
        avg_normal = np.mean(normals, axis=0)
        avg_normal /= np.linalg.norm(avg_normal)
        # Find all aunps with 25 of line through v in direction of avg_normal
        line_points = np.array([v + t * avg_normal for t in np.linspace(0, 50, 100)])
        idxs_aunps = tree_aunps.query_ball_point(line_points, cylinder_radius)
        # Generate list of unique inds in idxs_aunps
        unique_idxs_aunps = set()
        for idx_list in idxs_aunps:
            unique_idxs_aunps.update(idx_list)
        num_aunps_at_vertex.append(len(unique_idxs_aunps))

    v_array = np.array(subset_vertices)
    area_of_circle = np.pi * (25 ** 2)  # Area = πr²
    packing_coefficient = ((np.array(num_aunps_at_vertex)/2) * receptor_crosssection_nm_squared) / area_of_circle   

    return (v_array, packing_coefficient)