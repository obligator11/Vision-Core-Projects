# 👁️ Sayyam AI Lab: Real-Time Spatial Computing & Physics Suites

This repository houses a production-ready collection of cutting-edge computer vision systems, advanced kinematic pipelines, and hardware-accelerated interactive simulations. The core objective of these architectures is bridging high-level spatial mathematics (planar homography, tensor matrix manipulation, Delaunay triangulation) with zero-latency visual environments.

---

## 🛠️ Advanced Tech Stack & Optimization Engines

To bypass the Python Global Interpreter Lock (GIL) and maintain a flawless **60+ FPS** rendering loop during heavy deep learning inference, the codebase utilizes a completely decoupled asynchronous pipeline structure:

* **Core Vision Frameworks:** OpenCV (`cv2`) paired with MediaPipe Graph Topographies (Hands, FaceMesh, Pose 3D).
* **Object Tracking & ANPR Core:** YOLOv8 / YOLO11 Nano & Large configurations bridged via CUDA acceleration wrappers.
* **Newtonian Physics Engines:** Massive 2D rigid-body and elastic tracking steps handled via multi-jointed `PyMunk` spaces.
* **Inference Optimization:** Latent Consistency Models (LCM) and monocular depth pipes optimized via PyTorch CUDA tensors and TensorRT FP16 precision guidelines.
* **Concurrency Architecture:** Heavy inference loops offloaded to completely isolated OS worker daemons via lock-free Producer-Consumer multiprocessing queues (`multiprocessing.Process`).

---

## 📂 Repository Architecture

```text
Vision-Core_projects/
├── ar-combat-arcade/            # Hit-detection, vector reflection, & weapon metrics
│   ├── aegis_reflect.py         # AR Plasma Shield & Vector Deflection Engine
│   ├── holo_strike.py           # Kinematic Batting Engine with 3D Depth Scaling
│   ├── ouroboros.py             # Spatial 3D Snake Survival game loop
│   └── shinobi_slice.py         # AR Kinematic Slicing Engine (Barehand Tracking)
├── generative-ai-temporal/      # Local diffusion models & time-loop buffers
│   ├── prometheus.py            # Biomimetic Deep-Face Morphing engine
│   └── reality_warp.py          # Live background LCM restyling override
├── motion-biomechanics/         # Pose estimation, kinetic tracking, & velocity states
│   ├── kinesis_core.py          # Shadow-Boxing striking velocity tracker
│   ├── neon_rider.py            # Full-Body Rhythm & Flow Engine Aim Trainer
│   ├── red_light.py             # Masked Lucas-Kanade Optical Flow survival loop
│   └── spartan_core.py          # Biomechanical AI Coach & Telemetry HUD
├── physics-swarm-simulations/   # Unistroke tracking & parallel array metrics
│   └── stardust.py              # High-density vector point particle swarms
├── spatial-ui-holograms/        # Planar homography, 3D projections, & gesture control
│   ├── aero_canvas.py           # Kinematic Air-Drawing painting suite
│   ├── aether_piano.py          # Invisible Holographic Desk Homography Synthesizer
│   ├── atlas.py                 # Cranial 3D FaceMesh Euler angle translation
│   ├── bleeding_edge.py         # High-tension Bezier HUD ribbons & quantum layouts
│   ├── midas_core.py            # Monocular depth map calculation matrix
│   └── omni_grasp.py            # Kinematic OS Controller & Spatial Air-Mouse
└── surveillance-deep-tracking/  # Bounding regressions & temporal OCR trails
    ├── dynamic_cameraman.py     # Kalman-smoothed cinematic face framing crop
    ├── item_detector.py         # Deep tracking module via ByteTrack matching
    ├── overwatch_core.py        # Omniscient traffic enforcement ANPR engine
    └── sentinel.py              # Aegis Focus object-context监控 supervisor
