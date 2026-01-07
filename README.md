# Navi // The Wired (v23.0)

> *"No matter where you go, everyone's connected."*

**Navi** is a high-fidelity, immersive 3D network visualizer built for the **CodeSpring Hackathon**. Inspired by the aesthetic of *Serial Experiments Lain*, it transforms invisible digital infrastructure—packets, processes, and wireless signals—into a tangible, navigable cyberspace.

It is not just a dashboard; it is a flight simulator for your network interface.

---

## Visuals & Features

The application renders an infinite, procedurally generated tunnel representing the depth of the network. As you travel deeper, the environment shifts through distinct "Zones":

### The Environment
*   **Surface Layer:** Clean grid, standard system monitoring.
*   **The Sprawl:** High-density geometric architecture representing local infrastructure.
*   **Deep Web:** Green-tinted, heavy signal distortion, digital rain.
*   **The Blackwall:** A red, unstable firewall boundary protecting the core.
*   **Old Net:** Monochrome, static-laden ruins of legacy protocols.

### 📡 Data Entities
*   **Holographic Packet Stream:** Real-time network traffic visualized as high-velocity projectiles.
    *   **TCP:** Cyan Energy (Reliable, structured streams).
    *   **UDP:** Orange Plasma (Fast, fire-and-forget streams).
    *   **ICMP:** Magenta Beams (Ping/Diagnostic signals).
    *   **Live Metadata:** Each packet carries a floating holographic label displaying its Destination IP and Protocol.
*   **Process Orbitals:** Running system processes (`psutil`) rendered as satellites orbiting the central Hypercore.
*   **WiFi Beacons:** Nearby networks (`netsh`) detected and displayed as floating signal towers.
*   **Digital Rain:** Wall textures generated from the *actual hex payloads* of captured packets.

### Technical Aesthetics
*   **Post-Processing Pipeline:** Custom GLSL shaders for chromatic aberration, scanlines, and CRT distortion.
*   **Webcam Integration:** "Ghost" reflection of the user mapped onto the environment using OpenCV Canny edge detection.
*   **Spatial Audio:** Procedural drone and screech effects generated in real-time based on system load.

---

## System Architecture

The codebase has been refactored for **engineering excellence**, utilizing modern Python practices (Type Hinting, Dataclasses) and a clean separation of concerns.

*   **`main_gl.py`**: The Orchestrator. Manages the OpenGL context, Event Loop, and Shader Pipeline.
*   **`config.py`**: Centralized configuration for physics constants, colors, and zone thresholds.
*   **`entities.py`**: A modular collection of 3D objects (`PacketSystem`, `DigitalRain`, `Hypercore`) inheriting from a strictly typed `GameObject` base.
*   **`protocol.py`**: A robust, threaded packet sniffer using `scapy` and Python `dataclasses`.
*   **`wifi_scanner.py`**: Async wrapper for Windows native WiFi commands.

---

##  Installation

### Prerequisites
*   **OS:** Windows 10 / 11 (Required for Npcap & `netsh`).
*   **Python:** 3.10+
*   **Driver:** **[Npcap](https://npcap.com/)** (Install with **"WinPcap API-compatible mode"** checked).

### Setup
1.  **Clone the Uplink:**
    ```bash
    git clone https://github.com/TingleDinkle/Navi.git
    cd Navi
    ```

2.  **Initialize Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or manually:
    pip install pygame opencv-python scapy numpy PyOpenGL PyOpenGL_accelerate psutil
    ```

4.  **Jack In:**
    ```bash
    python main_gl.py
    ```

---

##  Controls

Navigate **The Wired** using standard First-Person flight controls:

*   **W / S:**  Fly Forward / Backward (Deep Dive)
*   **A / D:**  Strafe Left / Right
*   **Mouse:**  Look Around
*   **L-Click:** Terminate Process (Target a Satellite)
*   **ESC:**    Jack Out

---

##  Disclaimer

**Navi** acts as a passive network sniffer. It captures and visualizes traffic on your local network interface for educational and artistic purposes. 
*   **Privacy:** No data is stored or transmitted externally.
*   **Safety:** Ensure you have permission to monitor the network you are connected to.

---
*Status: Connected.