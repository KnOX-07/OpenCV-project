import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Define the lower and upper bounds for the red color in HSV
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on frame
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(frame, c, -1, (0, 255, 0), 2)

    cv2.imshow("Camera Preview", frame)
    # cv2.imshow("mask", mask)
    if cv2.waitKey(10) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
