import cv2
import numpy as np

# Adım 1: Resmi oku ve Canny uygula
img = cv2.imread("cember.jpg")   # kendi resminin adını yaz
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 10, 30)

# Adım 2: Median filtre
edges_median = cv2.medianBlur(edges, 3)
print("Kenar piksel sayısı:", np.count_nonzero(edges_median))

# Adım 3: HoughCircles
for p2 in [5, 8, 10, 15, 20, 30]:
    c = cv2.HoughCircles(edges_median, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                         param1=100, param2=p2, minRadius=10, maxRadius=300)
    print("param2 =", p2, "->", 0 if c is None else len(c[0]), "çember")

circles = cv2.HoughCircles(edges_median, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                           param1=100, param2=10, minRadius=10, maxRadius=300)

# Adım 4: Çemberleri çiz
result = img.copy()

if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0]:
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)   # çember: yeşil
        cv2.circle(result, (x, y), 2, (0, 0, 255), 3)   # merkez: kırmızı
    print("Bulunan çember sayısı:", len(circles[0]))
else:
    print("Çember bulunamadı")

# Adım 5: Göster ve kaydet
cv2.imshow("Canny", edges)
cv2.imshow("Median", edges_median)
cv2.imshow("HoughCircles", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("canny.jpg", edges)
cv2.imwrite("median.jpg", edges_median)
cv2.imwrite("circles.jpg", result)