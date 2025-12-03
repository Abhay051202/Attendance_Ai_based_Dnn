# AI_Based_Attendance_System

🚀 Project Setup & Installation Guide

Follow the steps below to set up the project environment and install all required dependencies.

✅ 1. Create a Virtual Environment

Use Python to create an isolated environment for the project:

python -m venv venv

✅ 2. Activate the Virtual Environment

On Windows PowerShell, run:

.\venv\Scripts\Activate.ps1


If you see a permission error, run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


Then activate again.

✅ 3. Install Project Dependencies

Install all necessary packages listed in requirements.txt:

pip install -r requirements.txt

✅ 4. Install InsightFace (Local Wheel File)

If the insightface-0.7.3-cp310-cp310-win_amd64.whl file is in your project directory, install it:

pip install insightface-0.7.3-cp310-cp310-win_amd64.whl

✅ 5. Verify Installation

To confirm that InsightFace and other packages are correctly installed:

pip show insightface
