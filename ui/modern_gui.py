import tkinter as tk
from tkinter import ttk, messagebox, Toplevel, filedialog
import threading
import queue
import cv2
from PIL import Image, ImageTk
import time
import math
from datetime import datetime, timedelta, date
import os
import csv
import sys
import re

# --- IMPORT BACKEND ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.database import DatabaseManager
from core.face_recognition import FaceRecognitionHandler
from core.attendance_tracker import AttendanceTracker
from core.video_processor import VideoProcessor
from core.registration import RegistrationModule
from core.utils import Utils, ThreadedCamera
from config.config import get_config

# --- THEME COLORS ---
COLORS = {
    'bg': '#1e1e2e', 'sidebar': '#181825', 'card': '#313244',
    'text': '#cdd6f4', 'accent': '#89b4fa', 'success': '#a6e3a1',
    'warning': '#f9e2af', 'danger': '#f38ba8', 'hover': '#45475a'
}

class ModernButton(tk.Button):
    def __init__(self, master, **kw):
        self.default_bg = kw.get('bg', COLORS['card'])
        self.hover_bg = kw.get('activebackground', COLORS['accent'])
        self.default_fg = kw.get('fg', COLORS['text'])
        self.hover_fg = '#1e1e2e'
        kw['relief'] = 'flat'; kw['borderwidth'] = 0; kw['cursor'] = 'hand2'
        kw['font'] = ("Segoe UI", 10, "bold")
        super().__init__(master, **kw)
        self.bind("<Enter>", self.on_enter); self.bind("<Leave>", self.on_leave)
    def on_enter(self, e): self['bg'] = self.hover_bg; self['fg'] = self.hover_fg
    def on_leave(self, e): self['bg'] = self.default_bg; self['fg'] = self.default_fg

class FaceAttendancePro:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance AI System")
        self.root.geometry("1150x700")
        self.root.configure(bg=COLORS['bg'])
        
        print("Initializing System...")
        self.db = DatabaseManager()
        self.face_handler = FaceRecognitionHandler(self.db)
        self.tracker = AttendanceTracker(self.db, self.face_handler)
        self.processor = VideoProcessor(self.face_handler)
        self.processor2 = VideoProcessor(self.face_handler)
        self.registrar = RegistrationModule(self.db, self.face_handler)
        
        self.caps = []
        self.is_running = False
        self.is_paused = False
        self.current_view = "dashboard"
        self.record_view_mode = "summary"
        
        self.camera_source_1 = tk.StringVar(value="Camera 0")
        self.camera_source_2 = tk.StringVar(value="Camera 1")
        
        self.scan_line_y = 0; self.scan_direction = 1; self.last_frame = None
        
        # Threading support
        self.processing_lock = threading.Lock()
        self.latest_results = {0: None, 1: None} # Stores (detections, labels, faces) for each camera index
        self.log_queue = queue.Queue()
        self.threads_running = False
        self.processing_threads = []

        self.setup_ui()
        self.animate_pulse()

    def setup_ui(self):
        self.sidebar = tk.Frame(self.root, bg=COLORS['sidebar'], width=220)
        self.sidebar.pack(side="left", fill="y"); self.sidebar.pack_propagate(False)
        
        try:
            img_path = "ui/prosper1.png"
            if not os.path.exists(img_path): img_path = "prosper1.png"
            if os.path.exists(img_path):
                pil = Image.open(img_path); w=200; h=int(float(pil.size[1])*(w/float(pil.size[0])))
                self.logo_img = ImageTk.PhotoImage(pil.resize((w,h), Image.Resampling.LANCZOS))
                tk.Label(self.sidebar, image=self.logo_img, bg=COLORS['sidebar']).pack(pady=(30,40))
            else: raise Exception
        except:
            tk.Label(self.sidebar, text="ATTENDANCE AI", bg=COLORS['sidebar'], fg=COLORS['accent'], font=("Segoe UI", 16, "bold")).pack(pady=(30,40))

        self.create_nav_btn("Dashboard", self.show_dashboard)
        self.create_nav_btn("New Registration", self.show_registration)
        self.create_nav_btn("Records & Reports", self.show_records)
        self.create_nav_btn("Exit System", self.close_app, color=COLORS['danger'])
        
        self.main_area = tk.Frame(self.root, bg=COLORS['bg'])
        self.main_area.pack(side="right", fill="both", expand=True, padx=15, pady=15)
        self.frame_dashboard = tk.Frame(self.main_area, bg=COLORS['bg'])
        self.frame_register = tk.Frame(self.main_area, bg=COLORS['bg'])
        self.frame_records = tk.Frame(self.main_area, bg=COLORS['bg'])
        
        self.setup_dashboard(); self.setup_registration(); self.setup_records()
        self.show_dashboard()

    def create_nav_btn(self, text, command, color=COLORS['card']):
        btn = ModernButton(self.sidebar, text=text, command=command, bg=COLORS['sidebar'], fg=COLORS['text'], activebackground=color, width=20)
        btn.pack(pady=8, padx=15, fill="x")

    def setup_dashboard(self):
        h = tk.Frame(self.frame_dashboard, bg=COLORS['bg']); h.pack(fill="x", pady=(0, 20))
        tk.Label(h, text="Dashboard", font=("Segoe UI", 24, "bold"), bg=COLORS['bg'], fg=COLORS['text']).pack(side="left")
        self.lbl_status = tk.Label(h, text="SYSTEM OFFLINE", font=("Segoe UI", 12, "bold"), bg=COLORS['bg'], fg=COLORS['danger']); self.lbl_status.pack(side="right")
        
        st = tk.Frame(self.frame_dashboard, bg=COLORS['bg']); st.pack(fill="x", pady=(0, 20))
        self.card_total = self.create_stat_card(st, "Total Registered", "0", "users")
        self.card_present = self.create_stat_card(st, "Present Today", "0", "check")
        
        c = tk.Frame(self.frame_dashboard, bg=COLORS['card'], padx=10, pady=10); c.pack(fill="x", pady=(0, 10))
        
        # Camera Options: Channels 1-16 (RTSP) + Webcams 0-4 (Local)
        cam_options = [f"Channel {i}" for i in range(1, 17)] + [f"Webcam {i}" for i in range(5)]
        
        tk.Label(c, text="Cam 1:", bg=COLORS['card'], fg=COLORS['text']).pack(side="left")
        self.camera_source_1 = tk.StringVar(value="Channel 1")
        self.cam_combo_1 = ttk.Combobox(c, textvariable=self.camera_source_1, values=cam_options, width=15); self.cam_combo_1.pack(side="left", padx=5)
        
        tk.Label(c, text="Cam 2:", bg=COLORS['card'], fg=COLORS['text']).pack(side="left", padx=(10,0))
        self.camera_source_2 = tk.StringVar(value="Channel 2")
        self.cam_combo_2 = ttk.Combobox(c, textvariable=self.camera_source_2, values=cam_options, width=15); self.cam_combo_2.pack(side="left", padx=5)
        
        self.btn_cam_toggle = ModernButton(c, text="START CAMERAS", command=self.toggle_camera, bg=COLORS['accent'], fg="#1e1e2e"); self.btn_cam_toggle.pack(side="right")
        self.btn_pause = ModernButton(c, text="PAUSE", command=self.toggle_pause, bg=COLORS['warning'], fg="#1e1e2e", state="disabled"); self.btn_pause.pack(side="right", padx=10)
        
        g = tk.Frame(self.frame_dashboard, bg=COLORS['bg']); g.pack(fill="both", expand=True)
        f1 = tk.Frame(g, bg="black"); f1.pack(side="left", fill="both", expand=True, padx=(0,5))
        self.video_label_1 = tk.Label(f1, bg="black", text="Camera 1 Off", fg="white"); self.video_label_1.pack(fill="both", expand=True)
        f2 = tk.Frame(g, bg="black"); f2.pack(side="left", fill="both", expand=True, padx=(5,0))
        self.video_label_2 = tk.Label(f2, bg="black", text="Camera 2 Off", fg="white"); self.video_label_2.pack(fill="both", expand=True)
        
        l = tk.Frame(self.frame_dashboard, bg=COLORS['card'], height=150); l.pack(fill="x", pady=(10, 0)); l.pack_propagate(False)
        tk.Label(l, text="Live Logs", bg=COLORS['card'], fg=COLORS['text'], font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=5, pady=2)
        self.log_list = tk.Listbox(l, bg=COLORS['card'], fg=COLORS['text'], relief="flat", highlightthickness=0); self.log_list.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.canvas_pulse = tk.Canvas(self.root, width=0, height=0); self.pulse_circle = self.canvas_pulse.create_oval(0,0,0,0)

    def create_stat_card(self, parent, title, value, icon):
        card = tk.Frame(parent, bg=COLORS['card'], padx=15, pady=15); card.pack(side="left", fill="x", expand=True, padx=5)
        tk.Label(card, text="●", font=("Segoe UI", 20), bg=COLORS['card'], fg=COLORS['accent']).pack(side="left", padx=(0, 15))
        info = tk.Frame(card, bg=COLORS['card']); info.pack(side="left")
        tk.Label(info, text=title, font=("Segoe UI", 9), bg=COLORS['card'], fg="#a6adc8").pack(anchor="w")
        lbl = tk.Label(info, text=value, font=("Segoe UI", 16, "bold"), bg=COLORS['card'], fg=COLORS['text']); lbl.pack(anchor="w")
        return lbl

    def setup_registration(self):
        tk.Label(self.frame_register, text="New Person Registration", font=("Segoe UI", 22, "bold"), bg=COLORS['bg'], fg=COLORS['text']).pack(anchor="w", pady=(0, 20))
        
        container = tk.Frame(self.frame_register, bg=COLORS['bg'])
        container.pack(fill="both", expand=True)

        # Left: Form
        form = tk.Frame(container, bg=COLORS['card'], padx=20, pady=20)
        form.pack(side="left", fill="y", padx=(0, 20))
        
        self.reg_entries = {}
        for field in ["Person ID (Unique)", "Full Name", "Email (Optional)", "Department (Optional)"]:
            tk.Label(form, text=field, bg=COLORS['card'], fg=COLORS['text']).pack(anchor="w", pady=(5, 2))
            e = tk.Entry(form, bg="#45475a", fg="white", relief="flat", font=("Segoe UI", 11))
            e.pack(fill="x", ipady=5); self.reg_entries[field] = e
            
        shift_fr = tk.Frame(form, bg=COLORS['card']); shift_fr.pack(fill="x", pady=15)
        tk.Label(shift_fr, text="Start:", bg=COLORS['card'], fg=COLORS['text']).pack(side="left")
        self.entry_shift_start = tk.Entry(shift_fr, bg="#45475a", fg="white", width=6, relief="flat"); self.entry_shift_start.insert(0,"09:00"); self.entry_shift_start.pack(side="left", padx=5)
        tk.Label(shift_fr, text="End:", bg=COLORS['card'], fg=COLORS['text']).pack(side="left", padx=(10,0))
        self.entry_shift_end = tk.Entry(shift_fr, bg="#45475a", fg="white", width=6, relief="flat"); self.entry_shift_end.insert(0,"18:00"); self.entry_shift_end.pack(side="left", padx=5)

        btn_fr = tk.Frame(form, bg=COLORS['card']); btn_fr.pack(fill="x", pady=20)
        ModernButton(btn_fr, text="CAPTURE & SAVE", command=self.perform_registration, bg=COLORS['success'], fg="#1e1e2e").pack(fill="x", pady=5)
        ModernButton(btn_fr, text="UPLOAD PHOTO", command=self.register_by_upload, bg=COLORS['accent'], fg="#1e1e2e").pack(fill="x")

        # Right: Preview
        preview_frame = tk.Frame(container, bg=COLORS['card'], padx=5, pady=5)
        preview_frame.pack(side="right", fill="both", expand=True)
        tk.Label(preview_frame, text="Captured Preview", bg=COLORS['card'], fg=COLORS['text'], font=("Segoe UI", 12, "bold")).pack(pady=(0, 5))
        self.reg_video_label = tk.Label(preview_frame, bg="black", text="No Image Captured", fg="white")
        self.reg_video_label.pack(fill="both", expand=True)

    def clear_registration_form(self):
        """Clears inputs after successful save"""
        for entry in self.reg_entries.values():
            entry.delete(0, 'end')
        
        self.entry_shift_start.delete(0, 'end'); self.entry_shift_start.insert(0, "09:00")
        self.entry_shift_end.delete(0, 'end'); self.entry_shift_end.insert(0, "18:00")
        
        self.reg_video_label.configure(image='', text="No Image Captured")
        self.reg_video_label.imgtk = None

    def setup_records(self):
        tk.Label(self.frame_records, text="Records & Reports", font=("Segoe UI", 22, "bold"), bg=COLORS['bg'], fg=COLORS['text']).pack(anchor="w", pady=(0, 15))
        
        top = tk.Frame(self.frame_records, bg=COLORS['bg']); top.pack(fill="x", pady=(0,10))
        self.btn_summary = ModernButton(top, text="Daily Summary", command=lambda: self.switch_record_view("summary"), bg=COLORS['accent'], fg="#1e1e2e", width=15); self.btn_summary.pack(side="left", padx=(0,5))
        self.btn_logs = ModernButton(top, text="Detailed Logs", command=lambda: self.switch_record_view("logs"), bg=COLORS['card'], fg=COLORS['text'], width=15); self.btn_logs.pack(side="left", padx=(0,5))
        self.btn_edit = ModernButton(top, text="Edit Database", command=lambda: self.switch_record_view("edit"), bg=COLORS['card'], fg=COLORS['text'], width=15); self.btn_edit.pack(side="left", padx=(0,5))

        search = tk.Frame(self.frame_records, bg=COLORS['card'], padx=5, pady=5); search.pack(fill="x", pady=5)
        tk.Label(search, text="Report ID:", bg=COLORS['card'], fg=COLORS['text']).pack(side="left")
        self.entry_search_id = tk.Entry(search, bg="#45475a", fg="white", relief="flat", width=10); self.entry_search_id.pack(side="left", padx=5)
        ModernButton(search, text="Generate Report", command=self.search_person_stats, bg=COLORS['accent'], fg="#1e1e2e").pack(side="left", padx=10)

        act = tk.Frame(self.frame_records, bg=COLORS['bg']); act.pack(fill="x", pady=5)
        ModernButton(act, text="Refresh", command=self.load_records, bg=COLORS['success'], fg="#1e1e2e", width=10).pack(side="left")
        ModernButton(act, text="Export CSV", command=self.export_csv, bg=COLORS['warning'], fg="#1e1e2e", width=10).pack(side="left", padx=5)
        self.btn_edit_sel = ModernButton(act, text="EDIT", command=self.open_edit_dialog, bg=COLORS['accent'], fg="#1e1e2e", width=10)
        self.btn_del_sel = ModernButton(act, text="DELETE", command=self.delete_selected_person, bg=COLORS['danger'], fg="#1e1e2e", width=10)

        style = ttk.Style(); style.theme_use("clam")
        style.configure("Treeview", background=COLORS['card'], fieldbackground=COLORS['card'], foreground=COLORS['text'], rowheight=25, borderwidth=0)
        style.configure("Treeview.Heading", background=COLORS['sidebar'], foreground=COLORS['text'], borderwidth=0, font=("Segoe UI", 10, "bold"))
        self.tree = ttk.Treeview(self.frame_records, show="headings", selectmode="browse"); self.tree.pack(fill="both", expand=True)
        
        # --- REPORT GENERATION PANEL ---
        report_frame = tk.Frame(self.frame_records, bg=COLORS['sidebar'], padx=10, pady=10)
        report_frame.pack(fill="x", pady=10)
        
        tk.Label(report_frame, text="Generate Reports", font=("Segoe UI", 12, "bold"), bg=COLORS['sidebar'], fg=COLORS['accent']).pack(anchor="w", pady=(0, 5))
        
        # Controls Row
        controls = tk.Frame(report_frame, bg=COLORS['sidebar'])
        controls.pack(fill="x")
        
        # Report Type
        tk.Label(controls, text="Type:", bg=COLORS['sidebar'], fg=COLORS['text']).pack(side="left")
        self.report_type = ttk.Combobox(controls, values=["Daily", "Weekly", "Monthly", "Yearly", "Custom"], state="readonly", width=10)
        self.report_type.current(0)
        self.report_type.pack(side="left", padx=5)
        self.report_type.bind("<<ComboboxSelected>>", self.update_report_dates)
        
        # Date Range
        tk.Label(controls, text="From:", bg=COLORS['sidebar'], fg=COLORS['text']).pack(side="left", padx=(10, 0))
        self.entry_date_start = tk.Entry(controls, bg=COLORS['card'], fg="white", width=12, relief="flat")
        self.entry_date_start.pack(side="left", padx=5)
        
        tk.Label(controls, text="To:", bg=COLORS['sidebar'], fg=COLORS['text']).pack(side="left")
        self.entry_date_end = tk.Entry(controls, bg=COLORS['card'], fg="white", width=12, relief="flat")
        self.entry_date_end.pack(side="left", padx=5)
        
        # Person Filter
        tk.Label(controls, text="Person ID:", bg=COLORS['sidebar'], fg=COLORS['text']).pack(side="left", padx=(10, 0))
        self.entry_report_id = tk.Entry(controls, bg=COLORS['card'], fg="white", width=10, relief="flat")
        self.entry_report_id.insert(0, "All")
        self.entry_report_id.pack(side="left", padx=5)
        
        # Buttons
        ModernButton(controls, text="PDF Report", command=lambda: self.generate_report("pdf"), bg=COLORS['danger'], fg="#1e1e2e", width=10).pack(side="right", padx=5)
        ModernButton(controls, text="CSV Report", command=lambda: self.generate_report("csv"), bg=COLORS['success'], fg="#1e1e2e", width=10).pack(side="right")

        self.update_report_dates(None) # Init dates
        
        self.switch_record_view("summary")

    def switch_record_view(self, mode):
        self.record_view_mode = mode
        for btn in [self.btn_summary, self.btn_logs, self.btn_edit]: btn.configure(bg=COLORS['card'], fg=COLORS['text'])
        self.btn_edit_sel.pack_forget(); self.btn_del_sel.pack_forget()
        
        if mode == "summary":
            self.btn_summary.configure(bg=COLORS['accent'], fg="#1e1e2e")
            cols = ("ID", "Name", "Login", "Logout", "Status")
        elif mode == "logs":
            self.btn_logs.configure(bg=COLORS['accent'], fg="#1e1e2e")
            cols = ("ID", "Name", "Date", "Time")
        elif mode == "edit":
            self.btn_edit.configure(bg=COLORS['accent'], fg="#1e1e2e")
            cols = ("ID", "Name", "Email", "Dept", "Shift Start", "Shift End")
            self.btn_edit_sel.pack(side="left", padx=5); self.btn_del_sel.pack(side="left", padx=5)
        
        self.tree["columns"] = cols
        for col in cols: self.tree.heading(col, text=col); self.tree.column(col, width=120)
        self.load_records()

    def load_records(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        if self.record_view_mode == "summary": recs = self.db.get_today_attendance()
        elif self.record_view_mode == "logs": recs = self.db.get_recent_logs()
        elif self.record_view_mode == "edit": recs = self.db.get_all_persons_details()
        for r in recs: self.tree.insert("", "end", values=r)

    def toggle_camera(self):
        if self.is_running:
            self.is_running = False
            self.threads_running = False # Stop background threads
            
            # Wait for threads to join (non-blocking in UI, but good practice to allow cleanup)
            # In a real GUI we might not want to block here, but for safety:
            for t in self.processing_threads:
                if t.is_alive(): t.join(timeout=0.2)
            self.processing_threads = []

            for cap in self.caps:
                if cap is not None: cap.release()
            self.caps = []
            
            self.video_label_1.config(image="", text="Camera 1 Off")
            self.video_label_2.config(image="", text="Camera 2 Off")
            self.cam_combo_1['state'] = 'readonly'
            self.cam_combo_2['state'] = 'readonly'
            self.btn_cam_toggle.config(text="START CAMERAS", bg=COLORS['accent'])
            self.btn_pause.config(state="disabled", text="PAUSE", bg=COLORS['warning']) # Reset pause button
            self.is_paused = False
            self.lbl_status.config(text="SYSTEM OFFLINE", fg=COLORS['danger'])
        else:
            print(f"Connecting Cameras...")
            # Try to connect to Camera 0 and Camera 1
            self.caps = []
            
            # Helper to update channel in RTSP URL
            def get_rtsp_url_with_channel(original_url, channel_num):
                if isinstance(original_url, str) and "rtsp" in original_url:
                    # Replace channel=X with channel=channel_num
                    new_url = re.sub(r'channel=\d+', f'channel={channel_num}', original_url)
                    print(f"Connecting to RTSP: ...{new_url[-20:]}") # Log last part for privacy/check
                    return new_url
                return original_url

            def get_source_from_selection(selection_str):
                """Parse dropdown selection to get video source"""
                try:
                    source = 0
                    if "Channel" in selection_str:
                        # RTSP Configured Camera
                        channel_num = selection_str.split(" ")[1]
                        base_url = get_config()['webcam_index']
                        source = get_rtsp_url_with_channel(base_url, channel_num)
                    elif "Webcam" in selection_str:
                        # Local Webcam
                        source = int(selection_str.split(" ")[1])
                    
                    # ALWAYS return ThreadedCamera for smooth UI
                    print(f"Opening ThreadedCamera({source})...")
                    return ThreadedCamera(source)
                        
                except Exception as e:
                    print(f"Error parsing source '{selection_str}': {e}")
                    return None

            # Camera 1
            val1 = self.camera_source_1.get()
            cap1 = get_source_from_selection(val1)
            
            if cap1 and cap1.isOpened(): self.caps.append(cap1)
            else: self.caps.append(None); print(f"Camera Source 1 ({val1}) failed")
            
            # Camera 2
            val2 = self.camera_source_2.get()
            cap2 = get_source_from_selection(val2)
            
            if cap2 and cap2.isOpened(): self.caps.append(cap2)
            else: self.caps.append(None); print(f"Camera Source 2 ({val2}) failed")

            if all(c is None for c in self.caps):
                messagebox.showerror("Error", "No cameras found"); return

            self.is_running = True
            
            # Start background processing threads
            self.threads_running = True
            self.latest_results = {0: None, 1: None}
            
            for i in range(len(self.caps)):
                if self.caps[i]:
                    t = threading.Thread(target=self.background_processing_loop, args=(i,))
                    t.daemon = True
                    t.start()
                    self.processing_threads.append(t)

            self.cam_combo_1['state'] = 'disabled'
            self.cam_combo_2['state'] = 'disabled'
            self.btn_cam_toggle.config(text="STOP CAMERAS", bg=COLORS['danger'])
            self.btn_pause.config(state="normal") # Enable pause button
            self.lbl_status.config(text="SYSTEM ONLINE", fg=COLORS['success'])
            self.update_video_loop()

    def background_processing_loop(self, cam_index):
        """Background thread to run heavy face recognition"""
        # Select processor
        processor = self.processor if cam_index == 0 else self.processor2
        cap = self.caps[cam_index]
        
        while self.threads_running and self.is_running:
            if cap is None: break
            
            # Read latest frame
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            # Run Heavy Processing
            # Returns: (detections, labels, faces, messages)
            try:
                detections, labels, faces, messages = processor.process_frame(
                    frame, 
                    mark_attendance_callback=self.tracker.process_recognized_face,
                    unknown_person_callback=self.tracker.process_unknown_person
                )
                
                # Queue messages for main thread
                for msg in messages:
                    self.log_queue.put(msg)
                
                # Store visualization data safely
                with self.processing_lock:
                    self.latest_results[cam_index] = (detections, labels, faces)
            except Exception as e:
                print(f"Processing Error Cam {cam_index}: {e}")
            
            # Small sleep to yield CPU if processing is super fast (prevents 100% CPU on empty loops)
            time.sleep(0.001)

    def toggle_pause(self):
        """Toggle the pause state of the dashboard video feed"""
        if not self.is_running: return
        
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.btn_pause.config(text="RESUME", bg=COLORS['success'])
            self.video_label_1.config(text="PAUSED")
            self.video_label_2.config(text="PAUSED")
        else:
            self.btn_pause.config(text="PAUSE", bg=COLORS['warning'])

    def update_video_loop(self):
        if not self.is_running: return
        
        # We only READ frames here for display. Processing happens in background.
        
        for i, cap in enumerate(self.caps):
            if cap is None: continue
            
            # 1. Get Instant Frame (Non-blocking)
            ret, frame = cap.read()
            if not ret or frame is None:
                if i == 0: self.video_label_1.configure(image="", text="No Signal")
                else: self.video_label_2.configure(image="", text="No Signal")
                continue

            # 2. Get Latest Processing Results (Instant)
            results = None
            with self.processing_lock:
                results = self.latest_results.get(i)
            
            # 3. Annotate Frame (Fast drawing)
            annotated_frame = frame
            
            if results:
                detections, labels, faces = results
                
                # Use the processor's drawing method
                processor = self.processor if i == 0 else self.processor2
                annotated_frame = processor.annotate_frame(frame, detections, labels, faces)
            
            # Flush Logs
            while not self.log_queue.empty():
                try:
                    msg = self.log_queue.get_nowait()
                    self.log_list.insert(0, f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
                except: break

            # 4. Display
            if not self.is_paused:
                img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                img.thumbnail((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                
                if i == 0:
                    self.video_label_1.imgtk = imgtk
                    self.video_label_1.configure(image=imgtk, text="")
                elif i == 1:
                    self.video_label_2.imgtk = imgtk
                    self.video_label_2.configure(image=imgtk, text="")

        # Update stats occasionally
        if int(time.time())%2==0 and self.current_view == "dashboard": 
            try:
                s = self.db.get_statistics()
                self.card_total.config(text=str(s['total_persons'])); self.card_present.config(text=str(s['present_today']))
            except: pass
                
        self.root.after(30, self.update_video_loop) # Target ~30 FPS

    def perform_registration(self):
        pid = self.reg_entries["Person ID (Unique)"].get()
        name = self.reg_entries["Full Name"].get()
        s_start = self.entry_shift_start.get()
        s_end = self.entry_shift_end.get()
        if not pid or not name: messagebox.showwarning("Missing", "ID/Name required"); return
        
        # 1. Open Cam, Snap, Close
        # Use Camera 0 for registration by default
        try: idx = 0 
        except: idx = 0
        temp_cap = cv2.VideoCapture(idx)
        if not temp_cap.isOpened(): messagebox.showerror("Error", "Could not open camera"); return
        for _ in range(5): ret, frame = temp_cap.read() # Warmup
        temp_cap.release()
        if not ret: messagebox.showerror("Error", "Failed capture"); return
        img = frame
        
        # 2. Show Preview
        prev = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        prev.thumbnail((400, 300))
        p_tk = ImageTk.PhotoImage(image=prev)
        self.reg_video_label.imgtk = p_tk; self.reg_video_label.configure(image=p_tk, text="")

        # 3. Extract & Save
        encoding, msg = self.face_handler.extract_face_encoding(img)
        if encoding is None: messagebox.showerror("Face Error", msg); return
        
        self.face_handler.load_face_encodings()
        exist_id, _, sim = self.face_handler.recognize_face(encoding)
        if exist_id and sim > 0.5: messagebox.showerror("Duplicate", f"Matches {exist_id}"); return

        success, db_msg = self.db.add_person(pid, name, encoding, self.reg_entries["Email (Optional)"].get(), self.reg_entries["Department (Optional)"].get(), s_start, s_end)
        if success: 
            self.face_handler.add_face_encoding(pid, name, encoding)
            messagebox.showinfo("Success", f"Registered {name}!")
            self.clear_registration_form() # Clear inputs
            self.show_dashboard()
        else: messagebox.showerror("Error", db_msg)

    def register_by_upload(self):
        pid = self.reg_entries["Person ID (Unique)"].get()
        name = self.reg_entries["Full Name"].get()
        s_start = self.entry_shift_start.get(); s_end = self.entry_shift_end.get()
        if not pid or not name: messagebox.showwarning("Missing", "ID/Name required"); return
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
        if not path: return
        img = cv2.imread(path)
        if img is None: messagebox.showerror("Error", "Read fail"); return
        
        # Preview
        prev = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        prev.thumbnail((400, 300))
        p_tk = ImageTk.PhotoImage(image=prev)
        self.reg_video_label.imgtk = p_tk; self.reg_video_label.configure(image=p_tk, text="")

        encoding, msg = self.face_handler.extract_face_encoding(img)
        if encoding is None: messagebox.showerror("Face Error", msg); return
        
        self.face_handler.load_face_encodings()
        exist_id, _, sim = self.face_handler.recognize_face(encoding)
        if exist_id and sim > 0.5: messagebox.showerror("Duplicate", f"Matches {exist_id}"); return

        success, db_msg = self.db.add_person(pid, name, encoding, self.reg_entries["Email (Optional)"].get(), self.reg_entries["Department (Optional)"].get(), s_start, s_end)
        if success: 
            self.face_handler.add_face_encoding(pid, name, encoding)
            messagebox.showinfo("Success", f"Registered {name}!")
            self.clear_registration_form() # Clear inputs
            self.show_dashboard()
        else: messagebox.showerror("Error", db_msg)

    def search_person_stats(self):
        pid = self.entry_search_id.get().strip()
        if not pid: messagebox.showwarning("Input", "Enter ID"); return
        data, msg = self.db.get_person_stats(pid)
        if not data: messagebox.showerror("Error", msg); return
        
        popup = Toplevel(self.root); popup.title(f"Report: {data['name']}"); popup.geometry("450x400"); popup.configure(bg=COLORS['bg'])
        tk.Label(popup, text=f"{data['name']}", font=("Segoe UI", 18, "bold"), bg=COLORS['bg'], fg=COLORS['accent']).pack(pady=(20, 5))
        tk.Label(popup, text=f"ID: {data['id']} | Shift: {data['shift']}", font=("Segoe UI", 10), bg=COLORS['bg'], fg="#a6adc8").pack(pady=(0, 20))
        stats = tk.Frame(popup, bg=COLORS['card'], padx=20, pady=20); stats.pack(fill="x", padx=20)
        def add_row(l, v, c):
            r = tk.Frame(stats, bg=COLORS['card']); r.pack(fill="x", pady=8)
            tk.Label(r, text=l, font=("Segoe UI", 11), bg=COLORS['card'], fg=COLORS['text']).pack(side="left")
            tk.Label(r, text=str(v), font=("Segoe UI", 11, "bold"), bg=COLORS['card'], fg=c).pack(side="right")
        add_row("Total Days:", data['total_days'], COLORS['text'])
        add_row("Late Arrivals:", data['late'], COLORS['warning'])
        add_row("Early Out:", data['early'], COLORS['danger'])
        add_row("Avg Hours:", f"{data['avg_hours']} hrs", COLORS['success'])

    def open_edit_dialog(self):
        sel = self.tree.selection()
        if not sel: return
        vals = self.tree.item(sel[0])['values']; pid = str(vals[0])
        popup = Toplevel(self.root); popup.geometry("400x500"); popup.configure(bg=COLORS['bg'])
        tk.Label(popup, text=f"Editing {pid}", font=("Segoe UI", 14, "bold"), bg=COLORS['bg'], fg=COLORS['accent']).pack(pady=15)
        entries = {}; labels = ["Name", "Email", "Department", "Shift Start", "Shift End"]; curr = [vals[1], vals[2], vals[3], vals[4], vals[5]]
        for i, lbl in enumerate(labels):
            tk.Label(popup, text=lbl, bg=COLORS['bg'], fg=COLORS['text']).pack(anchor="w", padx=40)
            ent = tk.Entry(popup, bg=COLORS['card'], fg="white", relief="flat"); ent.insert(0, str(curr[i])); ent.pack(fill="x", padx=40, pady=5); entries[lbl] = ent
        def save():
            if self.db.update_person(pid, entries["Name"].get(), entries["Email"].get(), entries["Department"].get(), entries["Shift Start"].get(), entries["Shift End"].get())[0]:
                messagebox.showinfo("OK", "Updated!"); self.load_records(); popup.destroy()
        ModernButton(popup, text="SAVE", command=save, bg=COLORS['success'], fg="#1e1e2e").pack(fill="x", padx=40, pady=30)

    def delete_selected_person(self):
        sel = self.tree.selection()
        if not sel: return
        pid = str(self.tree.item(sel[0])['values'][0])
        if messagebox.askyesno("Delete", "Delete person and logs?"):
            if self.db.delete_person(pid)[0]:
                self.face_handler.remove_face_encoding(pid)
                self.processor.clear_cache()
                messagebox.showinfo("OK", "Deleted")
                self.load_records()

    def export_csv(self): messagebox.showinfo("Export", self.db.export_to_csv(f"export_{datetime.now().strftime('%Y%m%d')}.csv")[1])

    def update_report_dates(self, event):
        """Auto-fill dates based on selection"""
        rtype = self.report_type.get()
        today = date.today()
        start = today
        end = today
        
        if rtype == "Daily":
            start = today
        elif rtype == "Weekly":
            start = today - timedelta(days=today.weekday()) # Start of week (Mon)
        elif rtype == "Monthly":
            start = today.replace(day=1)
        elif rtype == "Yearly":
            start = today.replace(month=1, day=1)
            
        # Update Entries
        self.entry_date_start.delete(0, 'end'); self.entry_date_start.insert(0, start.strftime('%Y-%m-%d'))
        self.entry_date_end.delete(0, 'end'); self.entry_date_end.insert(0, end.strftime('%Y-%m-%d'))
        
        # Enable/Disable based on Custom
        state = "normal" if rtype == "Custom" else "disabled"

    def generate_report(self, fmt):
        start = self.entry_date_start.get()
        end = self.entry_date_end.get()
        pid = self.entry_report_id.get().strip()
        if not pid: pid = "All"
        
        # Fetch Data
        data = self.db.get_attendance_report(start, end, pid)
        if not data:
            messagebox.showinfo("Report", "No records found for this period.")
            return

        # Create Reports Directory
        today_str = datetime.now().strftime('%Y-%m-%d')
        # Updated path for reports
        report_dir = os.path.join("data", "reports", today_str)
        
        if not os.path.exists(report_dir):
            try:
                os.makedirs(report_dir)
            except OSError as e:
                messagebox.showerror("Error", f"Could not create report directory: {e}")
                return

        # Generate File
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"Report_{pid}_{start}_to_{end}_{timestamp}"
        full_path = os.path.join(report_dir, filename)
        
        if fmt == "csv":
            full_path += ".csv"
            try:
                with open(full_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Date', 'Name', 'ID', 'Arrival', 'Leaving', 'Status'])
                    writer.writerows(data)
                messagebox.showinfo("Success", f"Saved: {full_path}")
                try: os.startfile(report_dir)
                except: pass
            except Exception as e:
                messagebox.showerror("Error", str(e))
                
        elif fmt == "pdf":
            full_path += ".pdf"
            success, msg = self.db.export_to_pdf(data, full_path, title=f"Attendance Report ({start} to {end})")
            if success: 
                messagebox.showinfo("Success", f"Saved: {full_path}")
                try: os.startfile(report_dir)
                except: pass
            else: messagebox.showerror("Error", msg)

    def animate_pulse(self):
        t = time.time() * 5; h = f"#{int(100+50*math.sin(t)):02x}ff{int(100+50*math.sin(t)):02x}"
        if self.is_running: self.canvas_pulse.itemconfig(self.pulse_circle, fill=h)
        else: self.canvas_pulse.itemconfig(self.pulse_circle, fill=COLORS['danger'])
        self.root.after(100, self.animate_pulse)

    def close_app(self): self.is_running = False; self.root.destroy()

    def show_dashboard(self): self.switch_frame(self.frame_dashboard, "dashboard")
    
    def show_registration(self):
        # Force stop camera when entering registration so button can take over
        if self.is_running: self.toggle_camera()
        self.switch_frame(self.frame_register, "register")
        
    def show_records(self): self.switch_frame(self.frame_records, "records")

    def switch_frame(self, frame, name):
        self.frame_dashboard.pack_forget(); self.frame_register.pack_forget(); self.frame_records.pack_forget()
        frame.pack(fill="both", expand=True); self.current_view = name
        if name == "records": self.switch_record_view("summary")

if __name__ == "__main__":
    root = tk.Tk(); app = FaceAttendancePro(root); root.protocol("WM_DELETE_WINDOW", app.close_app); root.mainloop()