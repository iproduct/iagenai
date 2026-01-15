import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


def moving_least_squares_smoothing(pcd, radius=0.05):
    """Applies local plane projection for smoothing (MLS)."""
    points = np.asarray(pcd.points)
    tree = KDTree(points)
    smoothed_points = np.copy(points)

    for i, point in enumerate(points):
        indices = tree.query_ball_point(point, radius)
        if len(indices) < 5: continue

        neighbors = points[indices]
        distances = np.linalg.norm(neighbors - point, axis=1)
        weights = np.exp(-(distances ** 2) / (radius ** 2))

        centroid = np.average(neighbors, axis=0, weights=weights)
        shifted = neighbors - centroid
        cov = shifted.T @ np.diag(weights) @ shifted

        _, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]

        dist_to_plane = np.dot(point - centroid, normal)
        smoothed_points[i] = point - dist_to_plane * normal

    smoothed_pcd = o3d.geometry.PointCloud()
    smoothed_pcd.points = o3d.utility.Vector3dVector(smoothed_points)
    return smoothed_pcd


def generate_mesh(pcd, depth=9):
    """Generates a mesh using Poisson Surface Reconstruction."""
    # 1. Estimate normals (required for Poisson)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(10)  # Ensures normals face the same way

    # 2. Run Poisson Surface Reconstruction
    # 'depth' controls mesh resolution; higher = more detail
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # 3. Clean up: Poisson often creates artifacts in low-density areas
    # We filter out vertices with low density
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


if __name__ == "__main__":
    # --- Execution ---
    # Create noisy sphere
    sphere_pcd = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).sample_points_uniformly(5000)
    noise = np.random.normal(0, 0.03, (5000, 3))
    sphere_pcd.points = o3d.utility.Vector3dVector(np.asarray(sphere_pcd.points) + noise)

    # 1. Apply MLS Smoothing
    smoothed_pcd = moving_least_squares_smoothing(sphere_pcd, radius=0.1)

    # 2. Generate Mesh
    mesh = generate_mesh(smoothed_pcd)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.7, 1.0])

    # 3. Visualize Result
    o3d.visualization.draw_geometries([mesh], window_name="Generated Mesh from MLS Cloud")
