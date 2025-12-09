# Navi v22.0 // The Wired (OpenGL Engine)

A cybersecurity visualization tool inspired by "Serial Experiments Lain", rendering your system's data within an immersive OpenGL environment.

## Features

-   **3D OpenGL Rendering:** Hardware-accelerated visualization of your system's data.
-   **GLSL Shader Post-Processing:** Applies real-time visual effects like:
    -   VHS Color Bleed & Chromatic Aberration
    -   Pixel Sorting & Tearing
    -   Bloom/Glow
    -   CRT Scanlines & Vignette
-   **First-Person Navigation:** Explore the "Wired" with standard FPS controls (WASD + Mouse Look).
-   **Diegetic User Interface:**
    -   **Transparent Tunnel:** Navigate a wireframe tunnel acting as your viewport.
    -   **Process Satellites:** Active applications orbit the central core as 3D cubes.
    -   **Wall Monitors:** Live CPU and RAM statistics displayed on the tunnel walls.
    -   **Hex Wall:** A scrolling waterfall of random hexadecimal data projected onto a tunnel wall.
-   **Webcam "Ghost" Reflection:** Your real-time Canny-edge-detected webcam feed is mapped onto the far end of the tunnel.
-   **Dynamic Data Integration:** Visualizes network packets (as flying lines) and active WiFi networks (as distant beacons).
-   **Intro Sequence:** A thematic startup animation ("CLOSE THE WORLD / OPEN THE NEXT").

## Prerequisites

-   Python 3.7+
-   A webcam 
-   Npcap 2

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**
    -   Windows: `venv\Scripts\activate`
    -   Mac/Linux: `source venv/bin/activate`

3.  **Install dependencies:**
    ```bash
    pip install pygame opencv-python mediapipe scapy numpy PyOpenGL PyOpenGL_accelerate pyperclip
    ```
    *(Note: Scapy may require additional setup or drivers like Npcap on Windows for full functionality.)*

## Usage

Run the OpenGL-powered main script:

```bash
python main_gl.py
```

## Controls

-   **Mouse:** Look around
-   **W / S:** Move Forward / Backward
-   **A / D:** Strafe Left / Right
-   **ESC**: Quit the application.
