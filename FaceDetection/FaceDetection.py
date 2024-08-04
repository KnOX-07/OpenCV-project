import cv2
import sys

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win = "Face Detector"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

net = cv2.dnn.readNetFromCaffe(r"C:\Users\user\OneDrive\Documents\OpenCV-project\FaceDetection\deploy.prototxt",
                               r"C:\Users\user\OneDrive\Documents\OpenCV-project\FaceDetection\res10_300x300_ssd_iter_140000.caffemodel")

# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB = False, crop = False)

    # Run model
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_bottom_left = int(detections[0, 0, i, 3] * frame_width)
            y_bottom_left = int(detections[0, 0, i, 4] * frame_height)
            x_top_right = int(detections[0, 0, i, 5] * frame_width)
            y_top_right = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame, (x_bottom_left, y_bottom_left), (x_top_right, y_top_right), (0, 0, 255))
            label = 'Confidence: %.4f' % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_bottom_left, y_bottom_left - label_size[1]),
                                (x_bottom_left + label_size[0], y_bottom_left + base_line),
                                (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x_bottom_left, y_bottom_left),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    cv2.imshow(win, frame)

source.release()
cv2.destroyWindow(win)
