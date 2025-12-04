import sys
import os
import json
from datetime import datetime

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.database import DatabaseManager

class AttendanceAPI:
    def __init__(self):
        self.db = DatabaseManager()

    # --- PERSON MANAGEMENT (CRUD) ---

    def create_person(self, person_id, name, face_encoding, email=None, department=None, shift_start="09:00", shift_end="18:00"):
        """
        Create a new person record.
        Returns: (success, message)
        """
        return self.db.add_person(person_id, name, face_encoding, email, department, shift_start, shift_end)

    def get_person(self, person_id):
        """
        Retrieve a person's details by ID.
        Returns: Dictionary or None
        """
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM persons WHERE person_id = %s", (person_id,))
            person = cursor.fetchone()
            return person
        except Exception as e:
            print(f"API Error: {e}")
            return None
        finally:
            conn.close()

    def get_all_persons(self):
        """
        Retrieve all registered persons.
        Returns: List of dictionaries
        """
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM persons")
            persons = cursor.fetchall()
            return persons
        except Exception as e:
            print(f"API Error: {e}")
            return []
        finally:
            conn.close()

    def update_person(self, person_id, name, email, department, shift_start, shift_end):
        """
        Update an existing person's details.
        Returns: (success, message)
        """
        return self.db.update_person(person_id, name, email, department, shift_start, shift_end)

    def delete_person(self, person_id):
        """
        Delete a person and their associated logs.
        Returns: (success, message)
        """
        return self.db.delete_person(person_id)

    # --- ATTENDANCE DATA ---

    def get_today_attendance(self):
        """
        Get attendance records for the current day.
        Returns: List of tuples/dictionaries (depending on DB implementation, DB returns tuples currently)
        """
        return self.db.get_today_attendance()

    def get_attendance_history(self, start_date, end_date, person_id=None):
        """
        Get attendance records for a specific date range.
        """
        return self.db.get_attendance_report(start_date, end_date, person_id)

    def get_statistics(self):
        """
        Get system statistics (Total Registered, Present Today).
        """
        return self.db.get_statistics()

    # --- UNKNOWN FACES ---
    
    def get_unknown_faces(self, limit=50):
        """
        Retrieve recent unknown face logs.
        """
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM unknown_faces ORDER BY timestamp DESC LIMIT %s", (limit,))
            logs = cursor.fetchall()
            return logs
        except Exception as e:
            print(f"API Error: {e}")
            return []
        finally:
            conn.close()

    def get_unknown_face(self, record_id):
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM unknown_faces WHERE id = %s", (record_id,))
            return cursor.fetchone()
        finally:
            conn.close()

    def delete_unknown_face(self, record_id):
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM unknown_faces WHERE id = %s", (record_id,))
            conn.commit()
            return True, "Deleted successfully"
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    # --- ATTENDANCE TABLE CRUD ---

    def create_attendance(self, person_id, date_str, arrival, leaving, status):
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO attendance (person_id, date, arrival_time, leaving_time, status)
                VALUES (%s, %s, %s, %s, %s)
            """, (person_id, date_str, arrival, leaving, status))
            conn.commit()
            return True, "Attendance created"
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    def get_attendance_by_id(self, record_id):
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM attendance WHERE id = %s", (record_id,))
            return cursor.fetchone()
        finally:
            conn.close()

    def update_attendance(self, record_id, arrival, leaving, status):
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE attendance 
                SET arrival_time=%s, leaving_time=%s, status=%s
                WHERE id=%s
            """, (arrival, leaving, status, record_id))
            conn.commit()
            return True, "Attendance updated"
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    def delete_attendance(self, record_id):
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM attendance WHERE id=%s", (record_id,))
            conn.commit()
            return True, "Attendance deleted"
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    # --- FACE LOGS CRUD ---

    def create_face_log(self, person_id, name, date_str, time_str):
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO face_logs (person_id, name, date, time)
                VALUES (%s, %s, %s, %s)
            """, (person_id, name, date_str, time_str))
            conn.commit()
            return True, "Log created"
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    def get_all_logs(self, limit=100):
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM face_logs ORDER BY id DESC LIMIT %s", (limit,))
            return cursor.fetchall()
        finally:
            conn.close()

    def get_face_log(self, record_id):
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM face_logs WHERE id=%s", (record_id,))
            return cursor.fetchone()
        finally:
            conn.close()

    def delete_face_log(self, record_id):
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM face_logs WHERE id=%s", (record_id,))
            conn.commit()
            return True, "Log deleted"
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()
