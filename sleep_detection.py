import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.22
SLEEP_FRAMES_LIMIT = 20

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    p1, p2, p3, p4, p5, p6 = [pt(i) for i in eye_indices]

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)

def get_face_box(landmarks, w, h):
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)

cap = cv2.VideoCapture(0)
face_data = {}

print("[INFO] Press 'q' to exit")

with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for i, face_landmarks in enumerate(results.multi_face_landmarks):

                landmarks = face_landmarks.landmark

                if i not in face_data:
                    face_data[i] = {"closed": 0, "sleep": False}

                left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                avg_ear   = (left_ear + right_ear) / 2.0

            
                if avg_ear < EAR_THRESHOLD:
                    face_data[i]["closed"] += 1
                else:
                    face_data[i]["closed"] = 0
                    face_data[i]["sleep"] = False

                if face_data[i]["closed"] >= SLEEP_FRAMES_LIMIT:
                    face_data[i]["sleep"] = True

                
                x1, y1, x2, y2 = get_face_box(landmarks, w, h)

            
                if face_data[i]["sleep"]:
                    color = (0, 0, 255)  
                    label = f"Student {i+1}: SLEEPING"
                else:
                    color = (0, 200, 0)  
                    label = f"Student {i+1}: AWAKE"

                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + 220, y1), color, -1)

            
                cv2.putText(frame, label,
                            (x1 + 5, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2)

        else:
            cv2.putText(frame, "No face detected",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2)

        cv2.imshow("Smart Classroom Monitor", frame)

        key = cv2.waitKey(1) & 0xFF

    
        if key == ord('q') or key == 27:
            break

cap.release()
cv2.destroyAllWindows()
