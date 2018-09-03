import dlib
from skimage import io
import numpy
import matplotlib.pyplot as plt
import cv2

# You should download this file manually
predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# feel free to use any photo you want
win = dlib.image_window()
img = io.imread('pics/person2.jpg')
# plt.imshow(img)

win.clear_overlay()
win.set_image(img)

# array of faces
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
abc = plt.figure(num=1, figsize=(8, 5))
for k, d in enumerate(dets):
    shape = predictor(img, d)
    np_shape = []
    for i in shape.parts():
        np_shape.append([i.x, i.y])
    np_shape = numpy.array(np_shape)

    plt.scatter(np_shape[:, 0], np_shape[:, 1], c='w', s=8)
    # plt.plot(shape_graphed_np[:, 0], shape_graphed_np[:, 1], c='w')
    print(np_shape)

    win.add_overlay(shape)  # 绘制特征点

    for idx, point in enumerate(np_shape):
        pos = (point[0], point[1])
        cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.3, color=(0, 255, 0))
        cv2.circle(img, pos, 3, color=(255, 255, 0))
    win.set_image(img)

# plt.show()
dlib.hit_enter_to_continue()