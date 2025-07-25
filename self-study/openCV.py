import cv2 as cv
import numpy as np

# Prepare object points
pattern_size = (7, 6)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Arrays for points
objpoints = []
imgpoints = []

# Load single image
img = cv.imread("../openCV-data/left01.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect corners
ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
if ret:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1),
                               (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    imgpoints.append(corners2)
else:
    raise RuntimeError("Failed to detect corners.")

# Call calibrateCamera on a single image
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Estimated intrinsic matrix K:")
print(mtx)
print("Estimated distortion coefficients:")
print(dist)
print("Reprojection error:", ret)

# Optional: undistort
h, w = img.shape[:2]
newcameramtx, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
undistorted = cv.undistort(img, mtx, dist, None, newcameramtx)
cv.imwrite("undistorted_singleview.png", undistorted)
sadf













# Ref: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob("../openCV-data/*.jpg")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (7,6), corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(500)

# cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread("../openCV-data/left12.jpg")
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('undistorted.png', dst)


# # undistort
# mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
 
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('remapped.png', dst)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(objpoints)) )