# Navi // The Wired (v23.0)

> "No matter where you go, everyone's connected."

**Navi** is a 3D network visualization engine designed to translate abstract digital infrastructure into a navigable, atmospheric environment. It maps live network packets, system processes, and wireless signals into a procedural cyberspace, offering a unique perspective on the data flowing through your machine.

---

## Visuals and Features

The application renders an infinite tunnel that represents the depth of the network. The environment evolves as you travel deeper through various zones.

### The Environment
*   **Surface Layer**: Clean grid with standard system monitoring.
*   **The Sprawl**: High-density geometry representing local infrastructure.
*   **Deep Web**: Green-tinted signal distortion and digital rain.
*   **The Blackwall**: A volatile red firewall boundary. Breach it by maintaining forward pressure against the system's resistance.
*   **Old Net**: The monochrome ruins of legacy protocols.
*   **The Ghost Room**: A final, hidden core reached after breaching the deepest layers. It contains a monolithic Macintosh 128k artifact and the presence of a "ghost" user.

### Data Entities
*   **Holographic Packet Stream**: Real-time traffic visualized as high-velocity projectiles. **TCP** is rendered in cyan, **UDP** in orange, and other protocols in grey.
*   **Process Orbitals**: System processes are rendered as satellites orbiting a central Hypercore. These can be terminated by clicking on them.
*   **WiFi Visualizer**: Nearby wireless networks are displayed as signal towers on the tunnel walls.
*   **Digital Rain**: Vertical data streams generated from the actual hex payloads of captured packets.

### Technical Implementation
*   **Shader Pipeline**: Custom GLSL shaders handle chromatic aberration, scanlines, and glitch effects.
*   **Webcam Reflection**: A "ghost" image of the user is processed via Canny edge detection and mapped onto the 3D environment.
*   **Spatial Audio**: Procedural drone and explosion effects that respond to system activity.

---

## System Architecture

The project is structured for modularity and performance:

| Component | Responsibility |
| :--- | :--- |
| `main_gl.py` | Manages the OpenGL context, camera physics, and the main simulation loop. |
| `config.py` | Central repository for colors, physics constants, and zone thresholds. |
| `entities.py` | Contains the object classes for all 3D geometry, from the infinite tunnel to the final Ghost Room. |
| `model_loader.py` | Handles the loading and optimization of GLB models into wireframe data. |
| `protocol.py` | A threaded packet sniffer utilizing `scapy` to monitor live traffic. |
| `wifi_scanner.py` | Interfaces with native system commands to track nearby wireless signals. |

---

## Installation

### Prerequisites
*   **OS**: Windows 10 or 11 (required for native WiFi and network commands).
*   **Python**: 3.10 or higher.
*   **Driver**: **Npcap** must be installed. Ensure "WinPcap API-compatible mode" is selected during installation.

### Setup
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/TingleDinkle/Navi.git
    cd Navi
    ```

2.  **Initialize a virtual environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install pygame opencv-python scapy numpy PyOpenGL PyOpenGL_accelerate psutil trimesh
    ```

4.  **Run the application**:
    ```bash
    python main_gl.py
    ```

---

## Controls

*   **W / S**: Fly Forward / Backward
*   **A / D**: Strafe Left / Right
*   **Mouse**: Look Around
*   **Left Click**: Terminate a targeted process satellite.
*   **ESC**: Exit the application.

---

## Technical Notes

Navi acts as a passive network sniffer for educational and artistic purposes. It does not store or transmit any captured data externally. Users should ensure they have the necessary permissions to monitor the network they are connected to.

**Status**: System operational. No exit found.
