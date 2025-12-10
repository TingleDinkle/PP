import threading
import subprocess
import time
import re

class WifiScanner(threading.Thread):
    def __init__(self):
        super().__init__()
        self.networks = [] # List of (SSID, Signal)
        self.running = True
        self.daemon = True

    def run(self):
        while self.running:
            try:
                # Windows command to list networks
                # Using check_output to get stdout
                output = subprocess.check_output(
                    ['netsh', 'wlan', 'show', 'networks', 'mode=bssid'], 
                    encoding='cp850', # Standard Windows console encoding
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                self._parse_output(output)
            except Exception as e:
                # print(f"WiFi Scan Error: {e}")
                pass
            
            time.sleep(20.0) # Scan every 20 seconds (scanning is slow)

    def _parse_output(self, output):
        # Very basic parsing
        current_ssid = "Unknown"
        found = []
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("SSID"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    current_ssid = parts[1].strip()
            elif line.startswith("Signal"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    signal_str = parts[1].strip().replace("%", "")
                    try:
                        signal = int(signal_str)
                        if current_ssid:
                            found.append((current_ssid, signal))
                    except ValueError:
                        pass
        
        # Update shared list atomically (assignment is atomic in Python)
        if found:
            self.networks = found

    def stop(self):
        self.running = False
