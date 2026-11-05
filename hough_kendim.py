import cv2
import numpy as np

# Adım 1: Resmi oku ve kenarları bul
img = cv2.imread("resimmmm.jpg")
img = cv2.resize(img, (400, 300))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

# Adım 2: Sin ve cos değerlerini baştan hesapla
thetas = np.deg2rad(np.arange(0, 180))
cos_t = np.cos(thetas)
sin_t = np.sin(thetas)

# Adım 3: Oylama matrisi
h, w = edges.shape
diag = int(np.ceil(np.sqrt(h**2 + w**2)))
accumulator = np.zeros((2 * diag, len(thetas)), dtype=np.int32)

# Adım 4: Oylama
ys, xs = np.nonzero(edges)
for x, y in zip(xs, ys):
    for t_idx in range(len(thetas)):
        rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag
        accumulator[rho, t_idx] += 1

# Adım 5: Oylama matrisini kaydet
acc_img = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("oylama_matrisi.png", acc_img)
cv2.imshow("Oylama Matrisi", acc_img)

# Adım 6: Eşik değere göre doğruları bul
threshold = 100
rho_idxs, theta_idxs = np.where(accumulator > threshold)

# Adım 7: Doğruları çiz
result = img.copy()
for r_idx, t_idx in zip(rho_idxs, theta_idxs):
    rho = r_idx - diag
    a, b = cos_t[t_idx], sin_t[t_idx]
    x0, y0 = a * rho, b * rho
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
    cv2.line(result, pt1, pt2, (0, 0, 255), 1)

cv2.imshow("Sonuc", result)
cv2.imwrite("sonuc.png", result)
cv2.waitKey(0)
cv2.destroyAllWindows()