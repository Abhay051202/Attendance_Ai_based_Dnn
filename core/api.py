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
