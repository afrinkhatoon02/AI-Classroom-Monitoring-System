import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

attention_score = 100

def get_landmark_points(landmarks, w, h, ids):
    points = []
    for id in ids:
        x = int(landmarks[id].x * w)
        y = int(landmarks[id].y * h)
        points.append((x, y))
    return points

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            # 👁️ Eye landmarks
            left_eye = get_landmark_points(face_landmarks.landmark, w, h, [33, 133])
            right_eye = get_landmark_points(face_landmarks.landmark, w, h, [362, 263])

            # 👄 Mouth landmarks
            mouth = get_landmark_points(face_landmarks.landmark, w, h, [13, 14])

            # 🧠 Head position (nose)
            nose = get_landmark_points(face_landmarks.landmark, w, h, [1])[0]

            # --- Eye direction check ---
            eye_center_x = (left_eye[0][0] + right_eye[1][0]) // 2
            if abs(eye_center_x - nose[0]) > 40:
                attention_score -= 1
                cv2.putText(frame, "Looking Side", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                cv2.putText(frame, "Looking Forward", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # --- Looking down ---
            if nose[1] > h * 0.6:
                attention_score -= 1
                cv2.putText(frame, "Looking Down", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # --- Talking detection ---
            mouth_open = abs(mouth[0][1] - mouth[1][1])
            if mouth_open > 15:
                attention_score -= 1
                cv2.putText(frame, "Talking", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Draw landmarks
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    # Clamp score
    attention_score = max(0, min(100, attention_score))

    cv2.putText(frame, f"Attention Score: {attention_score}%", (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Attention Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()