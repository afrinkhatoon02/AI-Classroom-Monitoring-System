from flask import Flask, request, jsonify, session
from flask_cors import CORS

from supabase import create_client
import subprocess
import sys
from config import SUPABASE_URL, SUPABASE_KEY

# ======================================================
# FLASK APP
# ======================================================

app = Flask(__name__)

app.secret_key = "faculty_secret_key"

CORS(app, supports_credentials=True)



# ======================================================
# SOCKET CONNECTION
# ======================================================



# ======================================================
# START AI FILES
# ======================================================





supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ======================================================
# HOME ROUTE
# ======================================================

@app.route('/home')
def home():
    return "Welcome to AI Based Smart Classroom"

# ======================================================
# FACULTY LOGIN
# ======================================================

@app.route('/login', methods=['POST'])
def login():

    data = request.json

    username = data.get('username')
    password = data.get('password')

    if not username or not password:

        return jsonify({
            'error': 'Username and Password required'
        }), 400

    try:

        response = (
            supabase.table("faculty_credentials")
            .select("username, faculty_id")
            .eq("username", username)
            .eq("password", password)
            .single()
            .execute()
        )

        user = response.data

        if user:

            # STORE SESSION
            session['faculty_username'] = username
            session['faculty_id'] = user['faculty_id']

            return jsonify({

                'message': 'Login Successful',

                'username': username,

                'faculty_id': user['faculty_id'],

                'redirect': '/dashboard'

            })

        else:

            return jsonify({
                'error': 'Invalid Credentials'
            }), 401

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500

# ======================================================
# LOGOUT
# ======================================================

@app.route('/logout', methods=['POST'])
def logout():

    session.pop('faculty_username', None)
    session.pop('faculty_id', None)

    return jsonify({
        'message': 'Logged Out Successfully'
    })

# ======================================================
# DASHBOARD CHECK
# ======================================================

@app.route('/dashboard')
def dashboard():

    if 'faculty_id' not in session:

        return jsonify({
            'error': 'Unauthorized Access'
        }), 401

    return jsonify({

        'message': 'Welcome Faculty',

        'faculty_id': session['faculty_id'],

        'faculty_username': session['faculty_username']

    })

# ======================================================
# GET FACULTY COURSES
# ======================================================

@app.route('/get_faculty_courses', methods=['GET'])
def get_faculty_courses():

    faculty_id = request.args.get('faculty_id')

    if not faculty_id:

        return jsonify({
            'error': 'Faculty ID required'
        }), 400

    try:

        # FETCH COURSES FROM faculty_allotted
        response = (

            supabase.table("faculty_allotted")

            .select("*")

            .eq("faculty_id", faculty_id)

            .execute()

        )

        allotted_courses = response.data

        if not allotted_courses:

            return jsonify({
                'error': 'No Courses Found'
            }), 404

        courses_data = []

        # FETCH COURSE DETAILS
        for item in allotted_courses:

            course_query = (

                supabase.table("courses")

                .select("course_name")

                .eq("course_id", item["course_id"])

                .single()

                .execute()

            )

            course_name = ""

            if course_query.data:

                course_name = course_query.data["course_name"]

            courses_data.append({

                "course_id": item["course_id"],

                "course_name": course_name,

                "section": item["section"],

                "dept": item["dept"],

                "semester": item["semester"]

            })

        return jsonify({

            "courses": courses_data

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ======================================================
# GET STUDENT INFO
# ======================================================

@app.route("/get_student_info", methods=["GET"])
def get_student_info():

    rollno = request.args.get("rollno")

    if not rollno:

        return jsonify({
            "error": "Roll Number Required"
        }), 400

    try:

        # STUDENT DETAILS
        student_query = (
            supabase.table("student")
            .select("*")
            .eq("rollno", rollno)
            .single()
            .execute()
        )

        if not student_query.data:

            return jsonify({
                "error": "Student Not Found"
            }), 404

        student = student_query.data

        # ATTENDANCE DETAILS
        attendance_query = (
            supabase.table("attendance")
            .select("*")
            .eq("rollno", rollno)
            .execute()
        )

        attendance_records = attendance_query.data

        # FETCH COURSE NAME
        for record in attendance_records:

            course_query = (
                supabase.table("courses")
                .select("course_name")
                .eq("course_id", record["course_id"])
                .single()
                .execute()
            )

            if course_query.data:

                record["course_name"] = course_query.data["course_name"]

            else:

                record["course_name"] = "Unknown"

        return jsonify({

            "student": student,

            "attendance": attendance_records

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ======================================================
# FACULTY REPORT
# ======================================================

@app.route("/get_faculty_info", methods=["GET"])
def get_faculty_info():

    course_id = request.args.get("course_id")
    dept = request.args.get("dept")
    semester = request.args.get("semester")
    section = request.args.get("section")

    if not all([course_id, dept, semester, section]):

        return jsonify({
            "error": "All parameters required"
        }), 400

    try:

        # =============================================
        # GET FACULTY
        # =============================================

        faculty_query = (
            supabase.table("faculty_allotted")
            .select("faculty_id")
            .eq("course_id", course_id)
            .eq("dept", dept)
            .eq("semester", semester)
            .eq("section", section)
            .execute()
        )

        if not faculty_query.data:

            return jsonify({
                "error": "Faculty not found"
            }), 404

        faculty_id = faculty_query.data[0]["faculty_id"]

        # =============================================
        # FACULTY DETAILS
        # =============================================

        faculty_details = (
            supabase.table("faculty")
            .select("*")
            .eq("id", faculty_id)
            .single()
            .execute()
        )

        faculty = faculty_details.data

        # =============================================
        # ATTENDANCE ABOVE 75
        # =============================================

        above_75 = (
            supabase.table("attendance")
            .select("*")
            .eq("course_id", course_id)
            .eq("section", section)
            .gt("percentage", 75)
            .execute()
        )

        # =============================================
        # ATTENDANCE BELOW 75
        # =============================================

        below_75 = (
            supabase.table("attendance")
            .select("*")
            .eq("course_id", course_id)
            .eq("section", section)
            .lte("percentage", 75)
            .execute()
        )

        attendance_above_75 = []
        attendance_below_75 = []

        # =============================================
        # FETCH STUDENT DETAILS
        # =============================================

        for entry in above_75.data:

            student_query = (
                supabase.table("student")
                .select("name, rollno, contact")
                .eq("rollno", entry["rollno"])
                .single()
                .execute()
            )

            if student_query.data:

                attendance_above_75.append({

                    "name": student_query.data["name"],

                    "rollno": student_query.data["rollno"],

                    "contact": student_query.data["contact"],

                    "percentage": entry["percentage"]

                })

        for entry in below_75.data:

            student_query = (
                supabase.table("student")
                .select("name, rollno, contact")
                .eq("rollno", entry["rollno"])
                .single()
                .execute()
            )

            if student_query.data:

                attendance_below_75.append({

                    "name": student_query.data["name"],

                    "rollno": student_query.data["rollno"],

                    "contact": student_query.data["contact"],

                    "percentage": entry["percentage"]

                })

        return jsonify({

            "faculty": {

                "faculty_id": faculty["id"],

                "name": faculty["name"],

                "contact": faculty["contact"],

                "image_url": faculty["image_url"]

            },

            "attendanceAbove75": attendance_above_75,

            "attendanceBelow75": attendance_below_75

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500
    # ======================================================
# GET REPORT DATA
# ======================================================

@app.route("/get_report_data", methods=["GET"])
def get_report_data():

    course_id = request.args.get("course_id")
    section = request.args.get("section")
    faculty_id = request.args.get("faculty_id")

    if not all([course_id, section, faculty_id]):

        return jsonify({
            "error": "Missing Parameters"
        }), 400

    try:

        # ============================================
        # GET ATTENDANCE DATA
        # ============================================

        attendance_query = (

            supabase.table("attendance")

            .select("*")

            .eq("course_id", course_id)

            .eq("section", section)

            .eq("faculty_id", faculty_id)

            .execute()

        )

        attendance_data = attendance_query.data

        students = []

        leaderboard = []

        for record in attendance_data:

            rollno = record["rollno"]

            # ============================================
            # STUDENT DETAILS
            # ============================================

            student_query = (

                supabase.table("student")

                .select("*")

                .eq("rollno", rollno)

                .single()

                .execute()

            )

            student = student_query.data

            # ============================================
            # TOTAL COUNT DETAILS
            # ============================================

            total_query = (

                supabase.table("total_count")

                .select("*")

                .eq("rollno", rollno)

                .single()

                .execute()

            )

            total = total_query.data

            if not total:

                total = {
                    "sleeping": 0,
                    "phone": 0,
                    "talking": 0,
                    "attentive": 0
                }

            student_data = {

                "name": student["name"],

                "rollno": rollno,

                "percentage": record["percentage"],

                "sleeping": total["sleeping"],

                "phone": total["phone"],

                "talking": total["talking"],

                "attentive": total["attentive"],

                "image_url": student["image_url"]

            }

            students.append(student_data)

            leaderboard.append(student_data)

        # ============================================
        # SORT LEADERBOARD
        # ============================================

        leaderboard = sorted(

            leaderboard,

            key=lambda x: x["attentive"],

            reverse=True

        )

        return jsonify({

            "students": students,

            "leaderboard": leaderboard

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ======================================================
# RUN APP
# ======================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="0.0.0.0",

        port=5000

    )