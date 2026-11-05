import cv2
import numpy as np

# Adım 1: Resmi oku
img = cv2.imread("resimmmm.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adım 2: Canny
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# Adım 3: Median filtre
edges_median = cv2.medianBlur(edges, 5)

# Adım 4: HoughLines
result1 = img.copy()
lines = cv2.HoughLines(edges_median, 1, np.pi / 180, 150)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(result1, pt1, pt2, (0, 0, 255), 2)

# Adım 5: HoughLinesP
result2 = img.copy()
linesP = cv2.HoughLinesP(edges_median, 1, np.pi / 180, 50,
                         minLineLength=50, maxLineGap=10)

if linesP is not None:
    for line in linesP:
        x1, y1, x2, y2 = line[0]
        cv2.line(result2, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Adım 6: Göster ve kaydet
cv2.imshow("Canny", edges)
cv2.imshow("Median", edges_median)
cv2.imshow("HoughLines", result1)
cv2.imshow("HoughLinesP", result2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("canny.jpg", edges)
cv2.imwrite("median.jpg", edges_median)
cv2.imwrite("houghlines.jpg", result1)
cv2.imwrite("houghlinesp.jpg", result2)