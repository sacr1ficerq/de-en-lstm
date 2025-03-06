# Simulate a harmless key press (e.g., F15) every 30 seconds
import time
import win32api
import win32con

import psutil
import os

# Set current process to high priority
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows-specific

while True:
    win32api.keybd_event(0x7E, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)  # F15 key (does nothing)
    win32api.keybd_event(0x7E, 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(15)