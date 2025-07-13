import cv2 as cv
import numpy as np
import glob
import os

# Directory containing images
img_dir = "./images/android/"
img_pattern = os.path.join(img_dir, "*.JPG")
image_files = glob.glob(img_pattern)

# Prepare object points
pattern_size = (9, 6)  # adjust to your checkerboard
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

best_K = None
best_dist = None
best_error = float("inf")
best_img = None
best_shape = None

def opencv_to_opengl_projection(K, width, height, znear=0.1, zfar=100.0):
    P = np.zeros((4, 4))
    P[0, 0] = 2.0 * K[0, 0] / width
    P[1, 1] = +2.0 * K[1, 1] / height  # ← flipped sign from - to + to match OpenGL's focal length
    P[2, 0] = 1.0 - 2.0 * K[0, 2] / width
    P[2, 1] = 2.0 * K[1, 2] / height - 1.0
    P[2, 2] = (zfar + znear) / (znear - zfar)
    P[2, 3] = -1.0
    P[3, 2] = 2.0 * zfar * znear / (znear - zfar)
    return P

for fname in image_files:
    img = cv.imread(fname)
    if img is None:
        print(f"Could not load {fname}")
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect corners
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
    if ret:
        corners2 = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        objpoints = [objp]
        imgpoints = [corners2]

        error, K, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        print(f"\n[{fname}]")
        print("Estimated intrinsic matrix K:\n", K)
        print("Estimated distortion coefficients:", dist.ravel())
        print("Reprojection error:", error)

        if error < best_error:
            best_error = error
            best_K = K.copy()
            best_dist = dist.copy()
            best_img = fname
            best_shape = gray.shape[::-1]

    else:
        print(f"Checkerboard NOT detected in {fname}")

if best_K is not None:
    width, height = best_shape
    P_opengl = opencv_to_opengl_projection(best_K, width, height)
    P_opengl = P_opengl.transpose()  # col major order for OpenGL

    print("\n==== Best result ====")
    print(f"Image: {best_img}")
    print("Smallest reprojection error:", best_error)
    print("K with smallest error:\n", best_K)
    print("Distortion coefficients:", best_dist.ravel())
    print("OpenGL projection matrix:\n", P_opengl.transpose())

    # Print as C++ code for glm::mat4
    print("\n// C++ code to set glm::mat4 projection_mat")
    for col in range(4):
        for row in range(4):
            val = P_opengl[row, col]
            print(f"projection_mat[{col}][{row}] = {val}f;")
else:
    print("No checkerboards detected in any images.")


# # Default
# 2.759 0.000 0.000 0.000
# 0.000 1.380 0.000 0.000
# 0.006 0.010 -1.002 -1.000
# 0.000 0.000 -0.200 0.000

# # My estimation
# 2.473 0.000 0.000 0.000
# 0.000 1.440 0.000 0.000
# 0.010 0.027 -1.002 -1.000
# 0.000 0.000 -0.200 0.000
