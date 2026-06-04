# =========================================================
# att.py
# =========================================================

from flask import Flask, request, jsonify
from flask_cors import CORS

from PIL import Image

import cv2
import face_recognition
import numpy as np
import os
import time

from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

from phone_detection import detect_phone

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

CORS(app)

# =========================================================
# FACE DIRECTORY
# =========================================================

KNOWN_FACES_DIR = "captured_faces"

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# =========================================================
# PHONE DETECTION CONTROL
# =========================================================

phone_detection_enabled = False

# =========================================================
# LOAD KNOWN FACES
# =========================================================

def load_known_faces():

    known_encodings = []
    known_rollnos = []

    for filename in os.listdir(KNOWN_FACES_DIR):

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            path = os.path.join(KNOWN_FACES_DIR, filename)

            try:

                image = cv2.imread(path)

                if image is None:

                    print(f"Cannot read: {filename}")
                    continue

                rgb_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                encodings = face_recognition.face_encodings(
                    rgb_image
                )

                if len(encodings) > 0:

                    known_encodings.append(
                        encodings[0]
                    )

                    rollno = os.path.splitext(filename)[0]

                    known_rollnos.append(rollno)

                    print(f"Loaded: {filename}")

                else:

                    print(f"No face found in {filename}")

            except Exception as e:

                print(f"Error loading {filename}: {e}")

    print(f"Total Faces Loaded: {len(known_encodings)}")

    return known_encodings, known_rollnos

# =========================================================
# UPDATE TOTAL CLASSES
# =========================================================

def update_total_classes(course_id, section, faculty_id):

    try:

        response = supabase.table("attendance") \
            .select("*") \
            .eq("course_id", course_id) \
            .eq("section", section) \
            .eq("faculty_id", faculty_id) \
            .execute()

        students = response.data

        for row in students:

            total = (row["total_classes"] or 0) + 1

            attended = row["classes_attended"] or 0

            unattended = total - attended

            percentage = round(
                (attended / total) * 100,
                2
            ) if total > 0 else 0

            supabase.table("attendance") \
                .update({

                    "total_classes": total,
                    "unattended_classes": unattended,
                    "percentage": percentage

                }) \
                .eq("id", row["id"]) \
                .execute()

        print("Total Classes Updated")

    except Exception as e:

        print("Error:", e)

# =========================================================
# MARK ATTENDANCE
# =========================================================

def mark_attendance(rollno, course_id, section, faculty_id):

    try:

        response = supabase.table("attendance") \
            .select("*") \
            .eq("rollno", rollno) \
            .eq("course_id", course_id) \
            .eq("section", section) \
            .eq("faculty_id", faculty_id) \
            .execute()

        if response.data:

            row = response.data[0]

            attended = (row["classes_attended"] or 0) + 1

            total = row["total_classes"] or 1

            unattended = total - attended

            percentage = round(
                (attended / total) * 100,
                2
            )

            supabase.table("attendance") \
                .update({

                    "classes_attended": attended,
                    "unattended_classes": unattended,
                    "percentage": percentage

                }) \
                .eq("id", row["id"]) \
                .execute()

            print(f"Attendance Marked : {rollno}")

        else:

            print(f"No attendance row found for {rollno}")

    except Exception as e:

        print(e)

# =========================================================
# START ATTENDANCE ROUTE
# =========================================================

@app.route("/start_attendance", methods=["POST"])
def start_attendance():

    try:

        data = request.json

        course_id = data.get("course_id")

        faculty_id = data.get("faculty_id")

        section = data.get("section")

        if not all([course_id, faculty_id, section]):

            return jsonify({

                "error": "Missing Fields"

            }), 400

        main(course_id, faculty_id, section)

        return jsonify({

            "message": "Attendance Started"

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 500

# =========================================================
# START PHONE DETECTION
# =========================================================

@app.route("/start_phone_detection", methods=["POST"])
def start_phone_detection():

    global phone_detection_enabled

    phone_detection_enabled = True

    return jsonify({

        "message": "Phone Detection Enabled"

    })

# =========================================================
# STOP PHONE DETECTION
# =========================================================

@app.route("/stop_phone_detection", methods=["POST"])
def stop_phone_detection():

    global phone_detection_enabled

    phone_detection_enabled = False

    return jsonify({

        "message": "Phone Detection Disabled"

    })

# =========================================================
# MAIN FUNCTION
# =========================================================

def main(course_id, faculty_id, section):

    global phone_detection_enabled

    known_encodings, known_rollnos = load_known_faces()

    if len(known_encodings) == 0:

        print("No Faces Found")

        return

    update_total_classes(course_id, section, faculty_id)

    marked_students = set()

    video_capture = cv2.VideoCapture(0)

    frame_width = int(video_capture.get(3))

    frame_height = int(video_capture.get(4))

    # =====================================================
    # ALIGN BOX
    # =====================================================

    box_size = 350

    box_x1 = (frame_width // 2) - (box_size // 2)

    box_y1 = (frame_height // 2) - (box_size // 2)

    box_x2 = box_x1 + box_size

    box_y2 = box_y1 + box_size

    while True:

        ret, frame = video_capture.read()

        if not ret:

            break

        rgb_frame = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )

        face_locations = face_recognition.face_locations(

            rgb_frame

        )

        face_encodings = face_recognition.face_encodings(

            rgb_frame,

            face_locations

        )

        # =====================================================
        # MULTIPLE FACES
        # =====================================================

        for face_encoding, face_location in zip(

            face_encodings,

            face_locations

        ):

            top, right, bottom, left = face_location

            # =================================================
            # FACE ALIGNMENT
            # =================================================

            margin = 10

            aligned = (

                left >= box_x1 - margin and
                right <= box_x2 + margin and
                top >= box_y1 - margin and
                bottom <= box_y2 + margin

            )

            if not aligned:

                cv2.putText(

                    frame,

                    "ALIGN FACE",

                    (40, 50),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1,

                    (0, 255, 255),

                    2

                )

                continue

            # =================================================
            # MATCH FACE
            # =================================================

            matches = face_recognition.compare_faces(

                known_encodings,

                face_encoding,

                tolerance=0.5

            )

            distances = face_recognition.face_distance(

                known_encodings,

                face_encoding

            )

            if len(distances) > 0:

                best_match_index = np.argmin(distances)

                if matches[best_match_index]:

                    rollno = known_rollnos[best_match_index]

                    # =========================================
                    # PHONE DETECTION
                    # =========================================

                    if phone_detection_enabled:

                        phone_detected = detect_phone(

                            frame,

                            left,

                            top,

                            right,

                            bottom,

                            rollno,

                            course_id,

                            section

                        )

                        # =====================================
                        # IF PHONE DETECTED
                        # DO NOT MARK ATTENDANCE
                        # =====================================

                        if phone_detected:

                            cv2.rectangle(

                                frame,

                                (left, top),

                                (right, bottom),

                                (0, 0, 255),

                                2

                            )

                            cv2.putText(

                                frame,

                                f"{rollno} PHONE",

                                (left, top - 10),

                                cv2.FONT_HERSHEY_SIMPLEX,

                                0.7,

                                (0, 0, 255),

                                2

                            )

                            continue

                    # =============================================
                    # MARK ONLY ONCE
                    # =============================================

                    if rollno not in marked_students:

                        mark_attendance(

                            rollno,

                            course_id,

                            section,

                            faculty_id

                        )

                        marked_students.add(rollno)

                    # =============================================
                    # SHOW ONLY ROLL NUMBER
                    # =============================================

                    cv2.rectangle(

                        frame,

                        (left, top),

                        (right, bottom),

                        (0, 255, 0),

                        2

                    )

                    cv2.putText(

                        frame,

                        rollno,

                        (left, top - 10),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.7,

                        (0, 255, 0),

                        2

                    )

                else:

                    cv2.rectangle(

                        frame,

                        (left, top),

                        (right, bottom),

                        (0, 0, 255),

                        2

                    )

                    cv2.putText(

                        frame,

                        "UNKNOWN",

                        (left, top - 10),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.7,

                        (0, 0, 255),

                        2

                    )

        # =====================================================
        # DRAW ALIGN BOX
        # =====================================================

        cv2.rectangle(

            frame,

            (box_x1, box_y1),

            (box_x2, box_y2),

            (255, 255, 0),

            2

        )

        cv2.putText(

            frame,

            "ALIGN FACE HERE",

            (box_x1, box_y1 - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 0),

            2

        )

        cv2.imshow(

            "Attendance System",

            frame

        )

        key = cv2.waitKey(1)

        if key == ord("q"):

            break

    video_capture.release()

    cv2.destroyAllWindows()

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5001
    )