# Navi v22.0 // The Wired (OpenGL Engine)
# Demo
https://www.youtube.com/watch?v=YISqZbKRxtc

A cybersecurity visualization tool inspired by "Serial Experiments Lain", rendering your system's data within an immersive OpenGL environment.

## Features

-   **3D OpenGL Rendering:** Hardware-accelerated visualization of your system's data.
-   **Visuals & Aesthetics:**
    -   **"Data Highway" Packet Stream:** Live network packets visualized as flowing laser beams organized by protocol (TCP/UDP), with readable floating payloads.
    -   **Digital Rain:** "Matrix"-style scrolling hexadecimal characters on the tunnel walls, powered by real captured packet data (Green/Red based on side).
    -   **Holographic Hypercore:** A complex, multi-layered rotating 3D construct in the center representing the CPU, featuring gyroscopic rings and orbiting data text.
    -   **System Bus Ring:** A neat orbital ring displaying active processes as glowing nodes with clear labels.
    -   **GLSL Shader Post-Processing:** Retro "Cyberpunk" aesthetic with scanlines, CRT vignette, and subtle color grading.
-   **First-Person Navigation:** Explore the "Wired" with standard FPS controls (WASD + Mouse Look).
-   **Diegetic User Interface:**
    -   **Transparent Tunnel:** Navigate a wireframe tunnel acting as your viewport.
    -   **Wall Monitors:** Live CPU and RAM statistics displayed as holographic signs.
    -   **WiFi Scanner:** Live detection of nearby WiFi networks, displayed as a list of glowing signal dots and SSIDs.
-   **Webcam "Ghost" Reflection:** Your real-time Canny-edge-detected webcam feed is mapped onto the far end of the tunnel.
-   **Performance Optimized:**
    -   Texture pre-caching for smooth text rendering.
    -   Batch rendering for particle systems.
    -   Framerate locking (60 FPS) for buttery smooth animations.

## Prerequisites

-   Python 3.7+
-   A webcam 
-   Npcap (Required for Scapy on Windows)

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
    pip install pygame opencv-python mediapipe scapy numpy PyOpenGL PyOpenGL_accelerate pyperclip psutil
    ```
    *(Note: Scapy requires Npcap on Windows for full functionality.)*

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
