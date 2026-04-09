import cv2
import mediapipe as mp
import time
import math
from ultralytics import YOLO

class Tracker:
    def __init__(self):
        self.faces = {}
        self.next_id = 0
        self.max_distance = 50

    def update(self, current_centroids, current_times):
        updated_faces = {}
        t = time.time()

        for cx, cy, mouth_open in current_centroids:
            best_id = None
            min_dist = float('inf')

            for fid, fdata in self.faces.items():
                px, py = fdata["centroid"]
                dist = math.hypot(cx - px, cy - py)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_id = fid

            if best_id is not None:
                fdata = self.faces.pop(best_id)
                fdata["centroid"] = (cx, cy)
                fdata["last_seen"] = t

                if mouth_open:
                    if fdata["talking_start"] is None:
                        fdata["talking_start"] = t
                else:
                    fdata["talking_start"] = None

                updated_faces[best_id] = fdata
            else:
                updated_faces[self.next_id] = {
                    "centroid": (cx, cy),
                    "talking_start": t if mouth_open else None,
                    "last_seen": t
                }
                self.next_id += 1

        self.faces = updated_faces
        return self.faces


def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def get_mar(landmarks):
    lip_top = landmarks[13]
    lip_bottom = landmarks[14]
    lip_left = landmarks[78]
    lip_right = landmarks[308]

    vertical = distance(lip_top, lip_bottom)
    horizontal = distance(lip_left, lip_right)

    if horizontal == 0:
        return 0
    return vertical / horizontal


def main():
    model = YOLO("yolov8n.pt")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5)

    tracker = Tracker()
    MAR_THRESHOLD = 0.05
    TALKING_THRESHOLD_SECONDS = 15

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        current_faces = []

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                mar = get_mar(face.landmark)
                is_open = mar > MAR_THRESHOLD

                nose = face.landmark[1]
                h, w, _ = frame.shape
                cx, cy = int(nose.x * w), int(nose.y * h)

                current_faces.append((cx, cy, is_open))

        tracked = tracker.update(current_faces, time.time())
        t = time.time()

        for fid, data in tracked.items():
            cx, cy = data["centroid"]
            start = data["talking_start"]

            if start:
                duration = t - start
                if duration > TALKING_THRESHOLD_SECONDS:
                    cv2.putText(frame, f"ID {fid} TALKING!", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                else:
                    cv2.putText(frame, f"ID {fid} talking...", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            else:
                cv2.putText(frame, f"ID {fid}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Talking Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()