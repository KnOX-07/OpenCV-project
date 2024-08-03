import cv2
import sys

s = 0   # Device index
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win = "Camera Preview"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: # Escape
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(win, frame)

source.release()
cv2.destroyWindow(win)
