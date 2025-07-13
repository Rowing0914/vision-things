import cv2
import numpy as np
from numpy.linalg import norm

# 1. Define camera intrinsics
K = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # no distortion

# 2. Ground-truth camera pose (rotation vector and translation vector)
rvec_true = np.array([[-0.0515883364], [0.0515883364], [-4.28748528e-16]])
tvec_true = np.array([[0.000816845162], [0.000816845162], [10.0691081]])

# 3. Define 3D object points in world coordinates
object_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0.5, 0.5, 1]
], dtype=np.float32)

# 4. Project 3D points to 2D using the ground-truth pose
image_points, _ = cv2.projectPoints(object_points, rvec_true, tvec_true, K, dist_coeffs)
image_points = image_points.squeeze()

# Optionally add small noise to simulate detection errors
noise = np.random.normal(0, 0.5, image_points.shape)
image_points_noisy = image_points + noise

# 5. Estimate camera pose from noisy 2D-3D correspondences using RANSAC
success, rvec_est, tvec_est, inliers = cv2.solvePnPRansac(
    object_points,
    image_points_noisy,
    K,
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)

if not success:
    print("Pose estimation failed.")
    exit()

# 6. Compute rotation error
R_true, _ = cv2.Rodrigues(rvec_true)
R_est, _ = cv2.Rodrigues(rvec_est)
R_diff = R_est @ R_true.T
angle_error_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
angle_error_deg = np.degrees(angle_error_rad)

# 7. Compute translation error
t_error = norm(tvec_est - tvec_true)

# 8. Reproject points using estimated pose to compute reprojection error
projected_estimated, _ = cv2.projectPoints(object_points, rvec_est, tvec_est, K, dist_coeffs)
projected_estimated = projected_estimated.squeeze()
reprojection_errors = norm(projected_estimated - image_points, axis=1)
avg_reproj_error = np.mean(reprojection_errors)

# 9. Print results
print("=== Pose Estimation Evaluation ===")
print("Rotation Error: {:.6f} degrees".format(angle_error_deg))
print("Translation Error: {:.6f} units".format(t_error))
print("Average Reprojection Error: {:.4f} pixels".format(avg_reproj_error))
print(rvec_est)
print(rvec_true)
print(tvec_est)
print(tvec_true)
print("Inliers:", inliers.ravel() if inliers is not None else "None")
