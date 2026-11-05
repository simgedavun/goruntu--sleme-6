import cv2
import numpy as np
test = np.full((200, 200, 3), 255, dtype=np.uint8)
for r in [20, 40, 60, 80]:
    cv2.circle(test, (100, 100), r, (0, 0, 0), 2)
cv2.imwrite("test_cember.png", test)