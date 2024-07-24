import cv2
import mediapipe as mp
import math

class poseDetector():

    def __init__(self, mode=False, complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        """
        Inisialisasi objek poseDetector.

        Parameters:
            mode (bool): Mode deteksi pose (default: False).
            complexity (int): Kompleksitas model (default: 1).
            smooth_landmarks (bool): Penyempurnaan landmark (default: True).
            enable_segmentation (bool): Aktifkan segmentasi (default: False).
            smooth_segmentation (bool): Penyempurnaan segmentasi (default: True).
            detectionCon (float): Confidence threshold untuk deteksi (default: 0.5).
            trackCon (float): Confidence threshold untuk tracking (default: 0.5).
        """
        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # Menginisialisasi modul mediapipe untuk menggambar landmark
        self.mpDraw = mp.solutions.drawing_utils
        # Menginisialisasi modul mediapipe untuk deteksi pose
        self.mpPose = mp.solutions.pose
        # Membuat objek pose detection
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        """
        Temukan pose dalam gambar.

        Parameters:
            img (numpy.ndarray): Gambar input.
            draw (bool): Apakah gambar akan digambar dengan pose terdeteksi (default: True).

        Returns:
            img (numpy.ndarray): Gambar dengan pose terdeteksi.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Proses deteksi pose menggunakan objek pose
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                # Gambar landmark dan skeleton pose
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
                
        return img

    def findPosition(self, img, draw=True):
        """
        Temukan posisi landmark dalam gambar.

        Parameters:
            img (numpy.ndarray): Gambar input.
            draw (bool): Apakah landmark akan digambar di atas gambar (default: True).

        Returns:
            lmList (list): Daftar landmark dengan format [id, cx, cy].
        """
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # Mendapatkan koordinat landmark dalam pixel
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    # Gambar landmark sebagai lingkaran pada gambar
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Hitung sudut antara tiga titik landmark dalam gambar.

        Parameters:
            img (numpy.ndarray): Gambar input.
            p1 (int): Indeks landmark pertama.
            p2 (int): Indeks landmark kedua.
            p3 (int): Indeks landmark ketiga.
            draw (bool): Apakah sudut akan digambar di atas gambar (default: True).

        Returns:
            angle (float): Sudut antara titik landmark.
        """
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        # Menghitung sudut antara tiga titik landmark
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - 
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle
        
        if draw:
            # Gambar garis antar landmark dan sudut pada gambar
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

def main():
    # Membuat objek poseDetector
    detector = poseDetector()
    # Membuka kamera
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            # Mendeteksi pose pada gambar
            img = detector.findPose(img)
            # Menampilkan gambar dengan pose terdeteksi
            cv2.imshow('Pose Detection', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # Melepas kamera dan menutup semua jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 