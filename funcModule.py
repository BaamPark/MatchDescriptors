import cv2
import random
import numpy as np
def mark_matching_points(img1, img2, kp1, kp2, distance_min_list, length_kp):
    kp1_coordinate = [p.pt for p in kp1]
    kp2_coordinate = [p.pt for p in kp2]

    kp1_list = []
    kp2_list = []
    for i in range(length_kp):
        kp1_list.append(kp1_coordinate[distance_min_list[i][1]])
        kp2_list.append(kp2_coordinate[distance_min_list[i][2]])

    b = 0
    for p in kp1_list:
        cv2.circle(img1, (int(p[0]), int(p[1])), radius=3, color=(0, 255, 0), thickness=-1)
        b += 1
        cv2.putText(img1, str(b), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    c = 0
    for p in kp2_list:
        cv2.circle(img2, (int(p[0]), int(p[1])), radius=3, color=(0, 255, 0), thickness=-1)
        c += 1
        cv2.putText(img2, str(c), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

def ransac(kp1_list, kp2_list, thresh):
    corr = [[kp1_list[i][0], kp1_list[i][1], kp2_list[i][0], kp2_list[i][1]] for i in range(len(kp1_list))]
    # print(corr)
    maxInliers = []
    finalH = None
    for i in range(1):
        #select four feature pairs at random
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        random_corr = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        random_corr = np.vstack((random_corr, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        random_corr = np.vstack((random_corr, corr4))

        h = calculateHomography(random_corr)
        inliers = []
        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            print("error is ", d)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        if len(maxInliers) > (len(corr) * thresh):
            break
        return finalH
def geometricDistance(correspondence, h):
    p1 = np.transpose(np.matrix([correspondence[0], correspondence[1], 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[2], correspondence[3], 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)
def calculateHomography(correspondences):
    A = []
    for i in range(4):
        x, y = correspondences[i][0], correspondences[i][1]
        u, v = correspondences[i][2], correspondences[i][3]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H
