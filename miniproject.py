import cv2
import numpy as np

sigma = 1

def apply_gaussian_blur(frame, sigma):
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(frame, (k, k), sigma)

cap = cv2.VideoCapture(0)

mode = 'o'  # default mode

def print_menu():
    print("\n=== Webcam Edge Detection Menu ===")
    print("o : Original frame")
    print("x : Sobel X")
    print("y : Sobel Y")
    print("m : Sobel Magnitude")
    print("s : Sobel + Threshold (bonus)")
    print("l : Laplacian of Gaussian")
    print("+ : Increase sigma (smoothing)")
    print("- : Decrease sigma")
    print("q : Quit program")
    print("=================================\n")

print_menu()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = apply_gaussian_blur(gray, sigma)

    if mode == 'o':
        output = frame
    elif mode == 'x':
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
        output = cv2.convertScaleAbs(sobelx)
    elif mode == 'y':
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
        output = cv2.convertScaleAbs(sobely)
    elif mode == 'm':
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        output = cv2.convertScaleAbs(magnitude)
    elif mode == 's':
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = cv2.convertScaleAbs(magnitude)
        _, output = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    elif mode == 'l':
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        output = cv2.convertScaleAbs(log)

    cv2.imshow("Webcam Edge Detection", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quit program pressed. Exiting...")
        break
    elif key in [ord('o'), ord('x'), ord('y'), ord('m'), ord('s'), ord('l')]:
        mode = chr(key)
        print(f"Mode changed: {mode}")
    elif key == ord('+'):
        sigma += 0.5
        print("Sigma increased:", sigma)
    elif key == ord('-'):
        sigma = max(0.5, sigma - 0.5)
        print("Sigma decreased:", sigma)

cap.release()
cv2.destroyAllWindows()
