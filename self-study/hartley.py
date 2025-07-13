import cv2
import numpy as np
from sympy import symbols, Matrix, simplify, solve, init_printing

init_printing()

# === STEP 1: Load the image ===
image = cv2.imread("3planes.png")  # <-- replace with your actual image path
if image is None:
    raise FileNotFoundError("Image file not found. Check the filename/path.")

# === OPTIONAL: View the image to pick points ===
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# === STEP 2: Replace this with your manually identified corners ===
# Format: list of 3 squares, each with 4 (x, y) points (same order as virtual ones)
left = [(63, 82), (265, 38), (272, 193), (103, 246)]
right = [(337, 42), (523, 111), (487, 272), (337, 196)]
middle = [(273, 229), (450, 276), (396, 435), (180, 360)]

image_points = [np.array(left, dtype=np.float32), np.array(right, dtype=np.float32), np.array(middle, dtype=np.float32),]

# === STEP 3: Define virtual square corners ===
virtual_pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)

# === STEP 4: Compute homographies ===
homographies = [cv2.findHomography(virtual_pts, pts)[0] for pts in image_points]

# === STEP 5: Extract circular points image projections ===
circular_pts_img = []
for H in homographies:
    h1, h2 = H[:, 0], H[:, 1]
    z_plus, z_minus = h1 + 1j * h2, h1 - 1j * h2
    circular_pts_img.extend([z_plus, z_minus])

# === STEP 6: Define symbolic ω ===
w11, w12, w22, w13, w23, w33 = symbols('w11 w12 w22 w13 w23 w33', real=True)
omega = Matrix([
    [w11, w12, w13],
    [w12, w22, w23],
    [w13, w23, w33]
])

# === STEP 7: Construct linear constraints ===
eqs = []
for H in homographies:
    h1, h2 = Matrix(H[:, 0]), Matrix(H[:, 1])
    eqs.append((h1.T * omega * h2)[0] == 0)
    eqs.append((h1.T * omega * h1)[0] - (h2.T * omega * h2)[0] == 0)

# === STEP 8: Solve the system ===
sol = solve(eqs, [w11, w12, w22, w13, w23], dict=True)  # leave w33 = 1

if not sol:
    print("No solution found.")
else:
    print("Solved ω (up to scale):")
    omega_sol = omega.subs(sol[0]).subs({w33: 1})
    print(omega_sol)

    # === STEP 9: Compute K from ω⁻¹ ===
    omega_inv = simplify(omega_sol.inv())
    print("K·Kᵗ = ω⁻¹:")
    print(omega_inv)

    # === Optional: Numerical K via Cholesky ===
    K_numeric = np.linalg.cholesky(np.array(omega_inv).astype(np.float64))
    print("Estimated intrinsic matrix K:")
    print(K_numeric)
