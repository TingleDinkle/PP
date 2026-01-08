# Navi // The Wired (v23.0)

> "No matter where you go, everyone's connected."

**Navi** is a high-fidelity network visualization engine built to translate abstract digital infrastructure into a navigable, atmospheric environment. It maps live network packets, system processes, and wireless signals into a procedural cyberspace, offering a unique perspective on the data flowing through your local machine.

It is designed not just as a tool, but as an immersive experience, a flight simulator for your network interface.

---

## Visuals and Features

The application renders an infinite tunnel representing the depth of the network. As you travel deeper, the environment shifts through distinct zones based on your logical distance from the surface.

### The Environment
*   **Surface Layer**: Clean grid with standard system monitoring.
*   **The Sprawl**: High-density geometric architecture representing local infrastructure.
*   **Deep Web**: Green-tinted signal distortion and digital rain.
*   **The Blackwall**: A volatile red firewall boundary. Breach it by maintaining forward pressure against the system's resistance.
*   **Old Net**: The monochrome ruins of legacy protocols.
*   **The Ghost Room**: The final hidden core. Contains a monolithic Macintosh 128k artifact and the presence of a "ghost" user.

### Data Entities
*   **Holographic Packet Stream**: Real-time traffic visualized as high-velocity projectiles. TCP is rendered in cyan, UDP in orange, and ICMP in magenta.
*   **Process Orbitals**: System processes are rendered as satellites orbiting a central Hypercore. These can be terminated by clicking on them.
*   **WiFi Visualizer**: Nearby wireless networks are displayed as signal towers on the tunnel walls, decoded via native Windows APIs.
*   **Digital Rain**: Vertical data streams generated from the actual hex payloads of captured packets.

### Technical Implementation
*   **Shader Pipeline**: Custom GLSL shaders handle chromatic aberration, scanlines, and CRT distortion.
*   **Webcam Reflection**: A "ghost" image of the user is processed via Canny edge detection and mapped onto the environment.
*   **Spatial Audio**: Procedural drone and explosion effects that respond to system activity.

---

## System Architecture

The codebase utilizes a component-based architecture to separate game logic from the rendering engine.

| Component | Responsibility |
| :--- | :--- |
| `main_gl.py` | The Engine. Manages the OpenGL context, windowing, and shader pipeline. |
| `world.py` | The Manager. Handles game state, player physics, and narrative logic. |
| `entities.py` | The Geometry. A collection of 3D objects inheriting from a modular GameObject base. |
| `wifi_scanner.py`| The Sensor. Interfaces with the native Windows WLAN API via ctypes. |
| `protocol.py` | The Sniffer. A threaded packet listener utilizing Scapy and Npcap. |
| `config.py` | The Source. Centralized constants for colors, thresholds, and physics. |

---

## Installation and Deployment

### Option 1: Portable Executable (Recommended for Demos)
1.  Install **[Npcap](https://npcap.com/)** (Ensure **"WinPcap API-compatible mode"** is checked).
2.  Run `dist/Navi.exe`.

### Option 2: Python Environment
1.  **Prerequisites**: Windows 10/11, Python 3.10+, and Npcap.
2.  **Setup**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Run**:
    ```bash
    python main_gl.py
    ```

---

## Controls

*   **W / S**: Fly Forward / Backward (Deep Dive)
*   **A / D**: Strafe Left / Right
*   **Mouse**: Look Around
*   **L-Click**: Terminate Process Satellite
*   **ESC**: Exit Simulation

---

**Status**: System operational. No exit found.