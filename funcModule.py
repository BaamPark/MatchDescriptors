import cv2

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
