import cv2
import numpy as np

# 1. Görüntü Yükleme ve Ön İşleme
image_path = "cember.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"'{image_path}' dosyası bulunamadı. Dosya yolunu kontrol edin.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=30, threshold2=100)
edges_median = cv2.medianBlur(edges, 3)

H, W = edges_median.shape

# 2. Parametreler ve Look-up Table (LUT)
r_min = 20
r_max = 120
r_step = 2
radii = np.arange(r_min, r_max, r_step)
num_radii = len(radii)

thetas = np.deg2rad(np.arange(0, 360, 2))
cos_thetas = np.cos(thetas)
sin_thetas = np.sin(thetas)

# 3. 3 Boyutlu Oylama Matrisi ve Standart Oylama İşlemi
accumulator = np.zeros((H, W, num_radii), dtype=np.uint32)
edge_pixels = np.argwhere(edges_median > 0)

print("--- Standart Cember Oylamasi Basliyor ---")
print(f"Kenar piksel sayisi: {len(edge_pixels)}")
print(f"Test edilen yaricap sayisi: {num_radii}")

for y, x in edge_pixels:
    for r_idx, r in enumerate(radii):
        a_coords = np.round(x - r * cos_thetas).astype(int)
        b_coords = np.round(y - r * sin_thetas).astype(int)

        valid = (a_coords >= 0) & (a_coords < W) & (b_coords >= 0) & (b_coords < H)
        accumulator[b_coords[valid], a_coords[valid], r_idx] += 1

print("3B oylama matrisi olusturuldu.")

# 4. Oylama Matrisinin Görsel Olarak Kaydedilmesi
projection_2d = np.max(accumulator, axis=2)
projection_norm = cv2.normalize(projection_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("oylama_matrisi_standart.png", projection_norm)
print("Oylama matrisi izdusumu 'oylama_matrisi_standart.png' olarak kaydedildi.")

# 5. Eşik Değere Göre Çember Tespiti ve Çizim
print("\n--- Cember Tespiti ---")
max_vote = np.max(accumulator)
threshold = int(max_vote * 0.45) if max_vote > 0 else 1

detected_circles = []
output_image = image.copy()

for r_idx, r in enumerate(radii):
    b_indices, a_indices = np.where(accumulator[:, :, r_idx] >= threshold)

    for b, a in zip(b_indices, a_indices):
        vote = accumulator[b, a, r_idx]

        too_close = False
        for (det_a, det_b, det_r, det_vote) in detected_circles:
            dist = np.hypot(a - det_a, b - det_b)
            if dist < 15 and abs(r - det_r) < 10:
                too_close = True
                break

        if not too_close:
            detected_circles.append((a, b, r, vote))

for a, b, r, vote in detected_circles:
    cv2.circle(output_image, (a, b), r, (0, 255, 0), 2)
    cv2.circle(output_image, (a, b), 2, (0, 0, 255), 3)

# 6. Sonuçların Kaydedilmesi ve Gösterilmesi
cv2.imwrite("cember_standart_sonuc.png", output_image)

print(f"Toplam tespit edilen standart cember sayisi: {len(detected_circles)}")
print("-" * 50)
print(f"{'No':<4} | {'Merkez (X, Y)':<15} | {'Yaricap (r)':<12} | {'Oy':<6}")
print("-" * 50)
for idx, (a, b, r, vote) in enumerate(detected_circles, 1):
    print(f"{idx:<4} | ({a}, {b}){'':<7} | {r} px{'':<6} | {vote}")
print("-" * 50)

cv2.imshow("Standart Cemberler", output_image)
cv2.imshow("Oylama Matrisi", projection_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()