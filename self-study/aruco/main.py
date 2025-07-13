import cv2
import numpy as np
import cv2.aruco as aruco

def draw_3d_cube(frame, rvec, tvec, cameraMatrix, distCoeff):
    """
    Draws a cube on the marker using pose estimation results.
    """
    # Define 3D points of a cube (in marker coordinate space)
    cube_3d = np.array([
        [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0], [-0.5, 0.5, 0],  # bottom face
        [0.5, 0.5, -1], [0.5, -0.5, -1], [-0.5, -0.5, -1], [-0.5, 0.5, -1]  # top face
    ])

    # Project to 2D image
    imgpts, _ = cv2.projectPoints(cube_3d, rvec, tvec, cameraMatrix, distCoeff)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw base
    cv2.drawContours(frame, [imgpts[:4]], -1, (0, 0, 255), 2)

    # Draw pillars
    for i in range(4):
        cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[i + 4]), (0, 0, 255), 2)

    # Draw top face
    cv2.drawContours(frame, [imgpts[4:]], -1, (255, 0, 0), 2)


# --- カメラの読込み ---
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    print("Camera not accessible")
    exit()

# --- カメラパラメータの定義 ---
size = frame.shape
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
fx, fy, cx, cy = focal_length, focal_length, center[0], center[1]
cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
distCoeff = np.zeros((4, 1))  # Assuming no lens distortion

# --- ArUco マーカー辞書と検出器の初期化 ---
dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(dict_aruco)

# --- メインループ ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    if len(corners) > 0:
        for points, id in zip(corners, ids):
            image_points_2D = np.array(points[0], dtype="double")  # 4x2
            figure_points_3D = np.array([  # 3D points in marker coordinate
                (0.5, 0.5, 0.0),
                (0.5, -0.5, 0.0),
                (-0.5, -0.5, 0.0),
                (-0.5, 0.5, 0.0),
            ])

            # 姿勢推定
            success, rvec, tvec = cv2.solvePnP(
                figure_points_3D,
                image_points_2D,
                cameraMatrix,
                distCoeff,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # マーカーの枠とID表示
                cv2.polylines(frame, [np.int32(points)], True, (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {id[0]}", tuple(np.int32(points[0][0])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # キューブ描画関数呼び出し
                draw_3d_cube(frame, rvec, tvec, cameraMatrix, distCoeff)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 終了処理 ---
cap.release()
cv2.destroyAllWindows()
