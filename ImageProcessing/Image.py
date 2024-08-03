import cv2
import matplotlib.pyplot as plt

cb_img = cv2.imread(r"C:\Users\user\OneDrive\Documents\OpenCV-project\ImageProcessing\checkerboard_18x18.png")
coke_img = cv2.imread(r"C:\Users\user\OneDrive\Documents\OpenCV-project\ImageProcessing\coca-cola-logo.png")

# Use matplotlib imshow()
plt.imshow(cb_img)
plt.title("Matplotlib imshow")
plt.show()

# Use OpenCV imshow(), display for 8 sec
window1 = cv2.namedWindow("w1")
cv2.imshow(window1, cb_img)
cv2.waitKey(8000)
cv2.destroyWindow(window1)

# Use OpenCV imshow(), display for 8 sec
window2 = cv2.namedWindow("w2")
cv2.imshow(window2, coke_img)
cv2.waitKey(8000)
cv2.destroyWindow(window2)

# Use OpenCV imshow(), display until any key is pressed
window3 = cv2.namedWindow("w3")
cv2.imshow(window3, cb_img)
cv2.waitKey(0)
cv2.destroyWindow(window3)

window4 = cv2.namedWindow("w4")
alive = True
while alive:
    # Use OpenCV imshow(), display until 'q' key is pressed
    cv2.imshow(window4, coke_img)
    keypress = cv2.waitKey(1)
    if keypress == ord('q'):
        alive = False
cv2.destroyWindow(window4)

cv2.destroyAllWindows()
stop = 1
