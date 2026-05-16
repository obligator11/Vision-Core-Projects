# 👁️ Sayyam AI Lab: Real-Time Spatial Computing & Physics Suites

This repository houses a production-ready collection of cutting-edge computer vision systems, advanced kinematic pipelines, and hardware-accelerated interactive simulations. The core objective of these architectures is bridging high-level spatial mathematics (planar homography, tensor matrix manipulation, Delaunay triangulation) with zero-latency visual environments.

---

## 🛠️ Advanced Tech Stack & Optimization Engines

To bypass the Python Global Interpreter Lock (GIL) and maintain a flawless **60+ FPS** rendering loop during heavy deep learning inference, the codebase utilizes a completely decoupled asynchronous pipeline structure:

* **Core Vision Frameworks:** OpenCV (`cv2`) paired with MediaPipe Graph Topographies (Hands, FaceMesh, Pose 3D).
* **Object Tracking & ANPR Core:** YOLOv8 / YOLO11 Nano & Large configurations bridged via CUDA acceleration wrappers.
* **Newtonian Physics Engines:** Massive 2D rigid-body and elastic tracking steps handled via multi-jointed `PyMunk` spaces.
* **Inference Optimization:** Latent Consistency Models (LCM) and monocular depth pipes optimized via PyTorch CUDA tensors and TensorRT FP16 precision guidelines.
* **Concurrency Architecture:** Heavy inference loops offloaded to completely isolated OS worker daemons via lock-free Producer-Consumer multiprocessing queues (`multiprocessing.Process` / `collections.deque`).

---

## 📂 Repository Architecture

```text
Computer-Vision/
├── ar-combat-arcade/            # Hit-detection, vector reflection, & weapon metrics
│   ├── AI AR Batting Game: Holo-Strike
│   ├── AI AR Slicing Game Engine
│   ├── AI Snake Game: Ouroboros Project
│   ├── AR Shield & Deflection Engine Script
│   ├── Building An AR Plasma Combat Simulator
│   └── Ronin: AR Katana Mesh Slicing Engine
├── motion-biomechanics/         # Pose estimation, kinetic tracking, & velocity states
│   ├── AI Biomechanical Coach: Project Spartan
│   ├── AI Shadow-Boxing: Kinesis Engine
│   ├── AI Survival Engine: Red-Light Protocol
│   ├── Building a Neon Rhythm Game
│   └── Kinetic AR Climbing Engine Build
├── spatial-ui-holograms/        # Planar homography, 3D projections, & gesture control
│   ├── AI Gesture-Controlled 3D Game
│   ├── AI Hand Tracking OS Controller
│   ├── AI Holographic Painting System Build
│   ├── AI Piano: Holographic Desk Synthesizer
│   ├── midas_core.py
│   ├── Project Looking Glass: AR Anchors
│   ├── Project Raijin: Hadouken VFX Script
│   ├── Project Rift: AR Dimensional Tear
│   └── Real-Time Cranial 3D Projection System
├── generative-ai-temporal/      # Local diffusion overrides & time-loop buffers
│   ├── AI Temporal Cloning Physics Simulation
│   ├── Live Generative AI Reality Warp
│   ├── prometheus.py
│   └── Real-Time AI Face Morphing System
├── surveillance-deep-tracking/  # Frame smoothing, context models, & OCR pipelines
│   ├── AI Focus Monitoring System Build
│   ├── AI Traffic Enforcement System Build
│   ├── Dynamic Cameraman AI Script
│   └── item detector
└── physics-swarm-simulations/   # Massively parallel coordinate matrices & symbols
    ├── AI AR Rune-Casting System Build
    ├── AI Ferrofluid Swarm & Symbiote
    ├── Python Elastic Magic Simulation
    ├── stardust
    └── Whiplash Engine: Hellfire Upgrade
