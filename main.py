import cv2
import cvzone  # cvzone version: 1.6.1
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot


cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plot_y = LivePlot(640, 360, [15, 50])

id_list = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratio_list = []
blink_counter = 0
counter = 0
colour = (250, 196, 2)


while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()

    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]
        for id in id_list:
            cv2.circle(img, face[id], 5, colour, cv2.FILLED)

        left_up, left_down = face[159], face[23]
        length_vertical, _ = detector.findDistance(left_up, left_down)
        cv2.line(img, left_up, left_down, (0, 200, 0), 3)

        left_left, left_right = face[130], face[243]
        length_horizontal, _ = detector.findDistance(left_left, left_right)
        cv2.line(img, left_left, left_right, (0, 200, 0), 3)

        ratio = (length_vertical / length_horizontal) * 100
        ratio_list.append(ratio)
        if len(ratio_list) > 3:
            ratio_list.pop(0)
        ratio_avg = sum(ratio_list) / len(ratio_list)

        if ratio_avg < 28 and counter == 0: # tune the number 28 according to how far or near you are
            blink_counter += 1
            colour = (0, 200, 0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                colour = (250, 196, 2)

        cvzone.putTextRect(img, f"Blink Counter: {blink_counter}", (50, 100), colorR=colour)

        img_plot = plot_y.update(ratio_avg, colour)
        img = cv2.resize(img, (640, 360))
        img_stack = cvzone.stackImages([img, img_plot], 1, 1)

    else:
        img = cv2.resize(img, (640, 360))
        img_stack = cvzone.stackImages([img, img], 1, 1)
    
    cv2.imshow("Image", img_stack)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
