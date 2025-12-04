from flask import Flask, request, jsonify
import sys
import os
import numpy as np

# Ensure we can import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.api import AttendanceAPI

app = Flask(__name__)
api = AttendanceAPI()

# --- ROUTES ---

@app.route('/api/persons', methods=['GET'])
def get_all_persons():
    persons = api.get_all_persons()
    return jsonify(persons)

@app.route('/api/person/<person_id>', methods=['GET'])
def get_person(person_id):
    person = api.get_person(person_id)
    if person:
        return jsonify(person)
    return jsonify({"error": "Person not found"}), 404

@app.route('/api/person', methods=['POST'])
def create_person():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required = ['id', 'name']
    if not all(k in data for k in required):
        return jsonify({"error": f"Missing required fields: {required}"}), 400

    # Handle face encoding (expecting list/array in JSON, converting to numpy)
    # For simplicity in this demo, we'll generate a random one if not provided, 
    # OR expect a list of 128 floats.
    if 'face_encoding' in data:
        encoding = np.array(data['face_encoding'])
    else:
        # In a real scenario, you'd upload an image, but for API data entry:
        encoding = np.random.rand(128) 

    success, msg = api.create_person(
        data['id'], 
        data['name'], 
        encoding, 
        data.get('email'), 
        data.get('department'),
        data.get('shift_start', '09:00'),
        data.get('shift_end', '18:00')
    )
    
    if success:
        return jsonify({"message": msg}), 201
    return jsonify({"error": msg}), 400

@app.route('/api/person/<person_id>', methods=['PUT'])
def update_person(person_id):
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    # We need current data to fill gaps if partial update, 
    # but api.update_person expects all fields.
    # Let's fetch current first.
    current = api.get_person(person_id)
    if not current:
        return jsonify({"error": "Person not found"}), 404
        
    success, msg = api.update_person(
        person_id,
        data.get('name', current['name']),
        data.get('email', current['email']),
        data.get('department', current['department']),
        data.get('shift_start', current['shift_start']),
        data.get('shift_end', current['shift_end'])
    )
    
    if success:
        return jsonify({"message": msg})
    return jsonify({"error": msg}), 400

@app.route('/api/person/<person_id>', methods=['DELETE'])
def delete_person(person_id):
    success, msg = api.delete_person(person_id)
    if success:
        return jsonify({"message": msg})
    return jsonify({"error": msg}), 400

@app.route('/api/attendance/today', methods=['GET'])
def get_today_attendance():
    records = api.get_today_attendance()
    # Convert tuples to dicts if needed, or just return as is (jsonify handles lists)
    return jsonify(records)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(api.get_statistics())

@app.route('/api/unknown', methods=['GET'])
def get_unknown_faces():
    return jsonify(api.get_unknown_faces())

if __name__ == '__main__':
    print("Starting API Server on http://localhost:5000")
    app.run(debug=True, port=5000)
