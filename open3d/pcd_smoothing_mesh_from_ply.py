import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


def moving_least_squares_smoothing(pcd, radius=0.05):
    """
    Applies MLS smoothing by projecting each point onto a locally fitted plane.
    """
    points = np.asarray(pcd.points)
    tree = KDTree(points)
    smoothed_points = np.copy(points)

    for i, point in enumerate(points):
        # 1. Find local neighbors within radius
        indices = tree.query_ball_point(point, radius)
        if len(indices) < 3:
            continue

        neighbors = points[indices]

        # 2. Compute weights (Gaussian weight based on distance)
        distances = np.linalg.norm(neighbors - point, axis=1)
        weights = np.exp(-(distances ** 2) / (radius ** 2))

        # 3. Weighted Least Squares Plane Fitting
        # Shift neighbors to local origin
        centroid = np.average(neighbors, axis=0, weights=weights)
        shifted_neighbors = neighbors - centroid

        # Weighted Covariance Matrix
        W = np.diag(weights)
        cov = shifted_neighbors.T @ W @ shifted_neighbors

        # The normal is the eigenvector with the smallest eigenvalue
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]

        # 4. Project the original point onto the local plane
        # Projection formula: p_proj = p - dot(p - centroid, normal) * normal
        dist_to_plane = np.dot(point - centroid, normal)
        smoothed_points[i] = point - dist_to_plane * normal

    # Create new Open3D point cloud
    smoothed_pcd = o3d.geometry.PointCloud()
    smoothed_pcd.points = o3d.utility.Vector3dVector(smoothed_points)
    smoothed_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
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
    # --- Usage Example ---
    # 1. Create a noisy sphere
    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).sample_points_uniformly(2000)
    # noise = np.random.normal(0, 0.02, (2000, 3))
    # sphere.points = o3d.utility.Vector3dVector(np.asarray(sphere.points) + noise)

    # 1. Read pointcloud from PLY file
    input_file = "out.ply"
    pcd = o3d.io.read_point_cloud(input_file)  # Read the point cloud

    # 2. Apply MLS
    smoothed_pcd = moving_least_squares_smoothing(pcd, radius=0.1)

    # 2. Generate Mesh
    mesh = generate_mesh(smoothed_pcd)
    mesh.compute_vertex_normals()
    # mesh.paint_uniform_color([0.5, 0.7, 1.0])

    # 3. Visualize
    # pcd.paint_uniform_color([1, 0.7, 0.7])  # Original (Red-ish)
    # smoothed_pcd.paint_uniform_color([0.7, 1, 0.7])  # Smoothed (Green-ish)
    o3d.visualization.draw_geometries([pcd, smoothed_pcd], window_name="Smoothed point cloud")

    # 3. Visualize Result
    o3d.visualization.draw_geometries([mesh], window_name="Generated Mesh from MLS Cloud")


