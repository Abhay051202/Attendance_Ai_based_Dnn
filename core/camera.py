import cv2
import threading
import time

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Buffer size 1 is crucial: it tells the camera to drop old frames
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.status = False
        self.frame = None
        self.stopped = False # Flag to stop the thread gracefully
        
        # Start the background thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                # Read the latest frame
                (status, frame) = self.capture.read()
                if status:
                    self.status = status
                    self.frame = frame
            # Tiny sleep to let other CPU tasks run
            time.sleep(0.01)

    def read(self):
        # Return the most recent frame found by the thread
        return self.status, self.frame
    
    def isOpened(self):
        return self.capture.isOpened()

    def release(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()
