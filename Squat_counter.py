import cv2
import numpy as np
import PoseModule as pm
import time

# Buka kamera
cap = cv2.VideoCapture(0)

# Buat objek detektor pose
detector = pm.poseDetector()

# Inisialisasi variabel untuk menghitung squat jump
count = 0
feedback = "Perbaiki Posisi"

# Inisialisasi variabel untuk mendeteksi loncatan kaki kanan dan kiri secara bergantian
prev_jump_side = None
prev_jump_time = time.time()

# Inisialisasi variabel untuk progress bar
progress_width = 400
progress_height = 20
progress_bar_color = (0, 255, 0)

# Loop utama untuk menangkap bingkai video dari kamera
while cap.isOpened():
    # Baca bingkai dari kamera
    ret, img = cap.read()  # 640 x 480
    
    # Temukan pose dalam bingkai
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    
    # Periksa apakah landmark terdeteksi
    if len(lmList) != 0:
        # Hitung sudut untuk evaluasi squat jump
        lututkanan = detector.findAngle(img, 23, 25, 27)
        pinggulkanan = detector.findAngle(img, 11, 23, 25)
        lututkiri = detector.findAngle(img, 24, 26, 28)
        pinggulkiri = detector.findAngle(img, 12, 24, 26)
        
        # Periksa loncatan kaki kanan
        if lututkanan <= 90 and pinggulkanan >= 150 and prev_jump_side != 'right':
            if prev_jump_side == 'left':
                count += 1
                feedback = "Lompat"
            prev_jump_side = 'right'
            prev_jump_time = time.time()
        
        # Periksa loncatan kaki kiri
        if lututkiri <= 90 and pinggulkiri >= 150 and prev_jump_side != 'left':
            if prev_jump_side == 'right':
                count += 1
                feedback = "Lompat"
            prev_jump_side = 'left'
            prev_jump_time = time.time()
        
        # Hitung progres bar
        progress = int(400 * min(1, (time.time() - prev_jump_time) / 2))  # Maksimum 3 detik antara lompatan
        
        # Gambar progres bar
        cv2.rectangle(img, (50, 150), (50 + progress, 150 + progress_height), progress_bar_color, cv2.FILLED)
        
        # Gambar hitungan squat jump
        cv2.putText(img, str(int(count)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        
        # Gambar umpan balik
        cv2.putText(img, feedback, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan bingkai dengan anotasi
    cv2.imshow('Hitungan Squat Jump', img)
    
    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Bebaskan kamera dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()