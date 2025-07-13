import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded images
main_image_path = "geeks-full.png"
sub_image_path = "geeks-half.jpg"

main_image = cv.imread(main_image_path)
sub_image = cv.imread(sub_image_path)

# Extend the existing FLANN matcher to include pose recovery from the essential matrix
def Flanned_Matcher_with_Pose(main_image, sub_image):
    sift = cv.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    key_point1, descr1 = sift.detectAndCompute(main_image, None)
    key_point2, descr2 = sift.detectAndCompute(sub_image, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descr1, descr2, k=2)

    good_matches = []
    pts1 = []
    pts2 = []

    # Ratio test
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            pts1.append(key_point1[m.queryIdx].pt)
            pts2.append(key_point2[m.trainIdx].pt)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    K = np.array([[800, 0, main_image.shape[1] / 2],
                  [0, 800, main_image.shape[0] / 2],
                  [0, 0, 1]], dtype=np.float64)

    if len(pts1) >= 5:
        pts1 = np.array(pts1, dtype=np.float64)
        pts2 = np.array(pts2, dtype=np.float64)

        E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, threshold=1.0, prob=0.999)
        inliers = int(mask.sum()) if mask is not None else 0

        # Recover relative pose
        retval, R, t, pose_mask = cv.recoverPose(E, pts1, pts2, K, mask=mask)
    else:
        inliers = 0
        R, t = None, None
        mask = None

    # Draw matches
    matches_mask = mask.ravel().tolist() if mask is not None else None
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matches_mask,
                       flags=0)

    output_img = cv.drawMatches(main_image, key_point1, sub_image, key_point2, good_matches, None, **draw_params)
    return output_img, inliers, R, t

# Run the updated pipeline
output_img_pose, inliers_found, R_est, t_est = Flanned_Matcher_with_Pose(main_image, sub_image)

# Save result
# output_path_pose = "output_with_pose.jpg"
# cv.imwrite(output_path_pose, output_img_pose)
print(inliers_found, R_est, t_est)
