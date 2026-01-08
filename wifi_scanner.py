import threading
import time
import ctypes
from ctypes import wintypes

# --- Native Windows WLAN API Structures ---
class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", wintypes.BYTE * 8)]

class WLAN_INTERFACE_INFO(ctypes.Structure):
    _fields_ = [("InterfaceGuid", GUID),
                ("strInterfaceDescription", wintypes.WCHAR * 256),
                ("isState", wintypes.DWORD)]

class WLAN_INTERFACE_INFO_LIST(ctypes.Structure):
    _fields_ = [("dwNumberOfItems", wintypes.DWORD),
                ("dwIndex", wintypes.DWORD),
                ("InterfaceInfo", WLAN_INTERFACE_INFO * 1)]

class DOT11_SSID(ctypes.Structure):
    _fields_ = [("uSSIDLength", wintypes.ULONG),
                ("ucSSID", wintypes.CHAR * 32)]

class WLAN_BSS_ENTRY(ctypes.Structure):
    _fields_ = [("dot11Ssid", DOT11_SSID),
                ("_pad", wintypes.BYTE * 4), # Alignment padding for 64-bit
                ("uPhyId", wintypes.ULONG),
                ("dot11BssPhyType", wintypes.DWORD),
                ("dot11BssType", wintypes.DWORD),
                ("uPhySpecificAttributes", wintypes.DWORD), # Placeholder for union
                ("lRssi", wintypes.LONG),
                ("uLinkQuality", wintypes.ULONG),
                ("bInRegDomain", wintypes.BOOLEAN),
                ("usBeaconPeriod", wintypes.USHORT),
                ("ullTimestamp", ctypes.c_ulonglong),
                ("ullHostTimestamp", ctypes.c_ulonglong),
                ("usCapabilityInformation", wintypes.USHORT),
                ("ulChCenterFrequency", wintypes.ULONG),
                # WLAN_RATE_SET: ULONG (4) + USHORT[126] (252) = 256 bytes
                ("wlanRateSet", wintypes.BYTE * 256), 
                ("ulIeOffset", wintypes.ULONG),
                ("ulIeSize", wintypes.ULONG)]

class WLAN_BSS_LIST(ctypes.Structure):
    _fields_ = [("dwTotalSize", wintypes.DWORD),
                ("dwNumberOfItems", wintypes.DWORD),
                ("wlanBssEntries", WLAN_BSS_ENTRY * 1)]

class WifiScanner(threading.Thread):
    def __init__(self):
        super().__init__()
        self.networks = [] # List of (SSID, Signal)
        self.running = True
        self.daemon = True
        
        # Load wlanapi.dll
        try:
            self.wlanapi = ctypes.windll.wlanapi
            self.client_handle = wintypes.HANDLE()
            self.negotiated_version = wintypes.DWORD()
            
            # 1 = Client Version 1 (XP), 2 = Client Version 2 (Vista+)
            result = self.wlanapi.WlanOpenHandle(
                2, None, ctypes.byref(self.negotiated_version), ctypes.byref(self.client_handle)
            )
            if result != 0:
                print(f"WlanOpenHandle failed: {result}")
                self.wlanapi = None
        except Exception as e:
            print(f"Failed to load wlanapi: {e}")
            self.wlanapi = None

    def run(self):
        if not self.wlanapi:
            return

        while self.running:
            try:
                self._scan_networks()
            except Exception as e:
                # print(f"Native WiFi Scan Error: {e}")
                pass
            
            time.sleep(5.0) # Faster updates possible with native API

    def _scan_networks(self):
        iface_list = ctypes.pointer(WLAN_INTERFACE_INFO_LIST())
        
        # 1. Enumerate Interfaces
        result = self.wlanapi.WlanEnumInterfaces(
            self.client_handle, None, ctypes.byref(iface_list)
        )
        if result != 0: return

        found_networks = {} 

        for i in range(iface_list.contents.dwNumberOfItems):
            p_iface = iface_list.contents.InterfaceInfo[i]
            p_guid = ctypes.byref(p_iface.InterfaceGuid)
            
            # 2. Get Network List (BSS)
            p_bss_list = ctypes.pointer(WLAN_BSS_LIST())
            result = self.wlanapi.WlanGetNetworkBssList(
                self.client_handle, p_guid, None, 0, None, None, ctypes.byref(p_bss_list)
            )
            
            if result == 0:
                # Iterate variable-length BSS entry array via address arithmetic
                base_addr = ctypes.addressof(p_bss_list.contents.wlanBssEntries)
                entry_size = ctypes.sizeof(WLAN_BSS_ENTRY)
                count = p_bss_list.contents.dwNumberOfItems
                
                for j in range(count):
                    entry = WLAN_BSS_ENTRY.from_address(base_addr + (j * entry_size))
                    
                    if entry.dot11Ssid.uSSIDLength > 0:
                        try:
                            ssid_len = entry.dot11Ssid.uSSIDLength
                            raw_ssid = entry.dot11Ssid.ucSSID[:ssid_len]
                            
                            # Robust Decoding Strategy
                            try:
                                ssid_str = raw_ssid.decode('utf-8')
                            except UnicodeDecodeError:
                                try:
                                    # Try Windows default encoding
                                    ssid_str = raw_ssid.decode('mbcs')
                                except Exception:
                                    # Latin-1 never fails (maps 1:1 bytes)
                                    ssid_str = raw_ssid.decode('latin-1')
                            
                            # Sanitization
                            # 1. Remove Null bytes (common in hidden networks)
                            ssid_str = ssid_str.replace('\x00', '')
                            
                            # 2. Check visibility
                            # If empty, whitespace only, or has unprintable control chars -> Hex Fallback
                            if not ssid_str or not ssid_str.strip() or not ssid_str.isprintable():
                                hex_str = raw_ssid.hex().upper()
                                if not hex_str: hex_str = "HIDDEN"
                                ssid_str = f"[{hex_str[:8]}]"
                            
                            signal = entry.uLinkQuality
                            
                            # Keep strongest signal for this SSID
                            if ssid_str not in found_networks or signal > found_networks[ssid_str]:
                                found_networks[ssid_str] = signal
                        except: pass
                
                self.wlanapi.WlanFreeMemory(p_bss_list)

        # Update Shared List
        if found_networks:
            self.networks = list(found_networks.items())
            self.networks.sort(key=lambda x: x[1], reverse=True)

    def stop(self):
        self.running = False
        if self.wlanapi and self.client_handle:
            self.wlanapi.WlanCloseHandle(self.client_handle, None)
