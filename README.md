# Navi v22.0 // The Wired (OpenGL Engine)

**A Real-time Cyberpunk Data Visualizer**

> "Present day, present time! Hahaha!"

Navi is an immersive 3D data visualization tool inspired by *Serial Experiments Lain* and classic cyberpunk aesthetics. It renders your local system's activity—processes, network traffic, and WiFi signals—as a navigable, procedural 3D world (The Wired).

Turn your boring task manager into a flight through cyberspace.

##  Visuals & Features

The application renders a continuous "infinite tunnel" representing the depth of the network. As you travel deeper, the environment changes:

*   **Zone System:**
    *   **Surface:** Clean grid, standard monitoring.
    *   **The Sprawl:** Increased density, distortion.
    *   **Deep Web:** Green-tinted, heavy glitch artifacts.
    *   **The Blackwall:** Red warning visuals, firewall boundaries.
    *   **Old Net:** High distortion, static, monochrome.

*   **Data Entities:**
    *   **Data Highway:** Live network packets (sniffed via `scapy`) rendered as laser beams. Colors indicate protocol (TCP vs UDP).
    *   **Process Orbitals:** Running system processes (`psutil`) visualized as satellites orbiting the central core.
    *   **WiFi Signals:** Nearby networks (`netsh`) detected and displayed as floating signal beacons.
    *   **Digital Rain:** Real-time hex dumps of packet payloads scrolling on the tunnel walls.
    *   **Holographic Hypercore:** A central rotating construct representing your CPU/System state.

*   **Aesthetics:**
    *   **Retro-Cyberpunk Shader:** Custom GLSL post-processing for scanlines, chromatic aberration, and CRT distortion.
    *   **Webcam Integration:** "Ghost" reflection of the user mapped onto the far end of the tunnel using Canny edge detection.

##  System Requirements

*   **OS:** Windows 10 / 11 (Required for `netsh` WiFi scanning and Npcap support).
*   **Python:** 3.7+
*   **Hardware:** Dedicated GPU recommended for smooth 60fps OpenGL rendering.
*   **Network Driver:** **[Npcap](https://npcap.com/)** (Install with "WinPcap API-compatible mode" checked). This is *mandatory* for packet sniffing to work.

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install pygame opencv-python scapy numpy PyOpenGL PyOpenGL_accelerate psutil
    ```

##  Usage

1.  Ensure you have installed **Npcap**.
2.  Run the main engine:
    ```bash
    python main_gl.py
    ```

##  Controls

Navigate "The Wired" using standard First-Person controls:

*   **W / S:** Fly Forward / Backward
*   **A / D:** Strafe Left / Right
*   **Mouse:** Look around
*   **ESC:** Jack out (Quit)

##  Architecture

For developers interested in the code structure:

*   **`main_gl.py`**: The core engine. Handles the Pygame/OpenGL initialization, main game loop, GLSL shaders, and the `SystemMonitor` thread.
*   **`entities.py`**: Contains all visual classes (`PacketSystem`, `CyberCity`, `DigitalRain`, etc.) that render the 3D objects.
*   **`protocol.py`**: Runs a background thread using `scapy` to sniff network packets and queue them for the visualizer.
*   **`wifi_scanner.py`**: A Windows-specific module that wraps `netsh wlan` commands to discover local WiFi networks.

##  Disclaimer

This tool performs active packet sniffing on your local network interface for visualization purposes. Ensure you have permission to monitor the network you are connected to.

---
*Status: Connected.*