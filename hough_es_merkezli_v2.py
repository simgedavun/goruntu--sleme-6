import cv2
import numpy as np

# 1. Resmi oku ve kenarları bul
img = cv2.imread("test_cember.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
h, w = edges.shape

# 2. Sin ve cos değerlerini baştan hesapla
thetas = np.deg2rad(np.arange(0, 360, 2))
cos_t = np.cos(thetas)
sin_t = np.sin(thetas)

# 3. 3 boyutlu oylama matrisi [b, a, r]
r_min, r_max = 10, 90
n_r = r_max - r_min + 1
accumulator = np.zeros((h, w, n_r), dtype=np.int32)

# 4. Oylama
ys, xs = np.nonzero(edges)
print("Kenar piksel sayisi:", len(xs))

for x, y in zip(xs, ys):
    for r_idx in range(n_r):
        r = r_min + r_idx
        a = np.round(x - r * cos_t).astype(int)
        b = np.round(y - r * sin_t).astype(int)
        valid = (a >= 0) & (a < w) & (b >= 0) & (b < h)
        np.add.at(accumulator, (b[valid], a[valid], r_idx), 1)

print("Matris boyutu:", accumulator.shape, "En yuksek oy:", accumulator.max())

# 5. Oylama matrisini resim olarak kaydet
proj = accumulator.max(axis=2)
proj_img = cv2.normalize(proj, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("oylama_matrisi_cember_v2.png", proj_img)

# 6. Eşiğe göre çemberleri bul (başka işlem yok)
threshold = 0.7 * len(thetas)
b_idx, a_idx, r_idx = np.where(accumulator > threshold)
print("Bulunan cember sayisi:", len(r_idx))
for b, a, r_i in zip(b_idx, a_idx, r_idx):
    print("merkez:", (a, b), "yaricap:", r_min + r_i, "oy:", accumulator[b, a, r_i])

# 7. Çiz
result = img.copy()
for b, a, r_i in zip(b_idx, a_idx, r_idx):
    cv2.circle(result, (int(a), int(b)), int(r_min + r_i), (0, 0, 255), 1)

cv2.imwrite("sonuc_cember_v2.png", result)
cv2.imshow("Oylama (r ekseni uzerinden max)", proj_img)
cv2.imshow("Sonuc", result)
cv2.waitKey(0)
cv2.destroyAllWindows()