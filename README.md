# Edge Client

Native Windows computer vision client for a biometric attendance research project.
Built with **C++20**, **Windows ML / DirectML**, and **Direct3D 11/12**, it provides
camera capture, face detection and tracking, presentation attack detection (PAD),
and face recognition.

This repository contains the edge runtime, diagnostic tools, and tests. The
frontend and backend are maintained in separate repositories.

## Project status

The project is an active research prototype. Individual vision components and
attendance orchestration are implemented; application integration and broader
evaluation remain in progress.

| Component | Current implementation |
|---|---|
| Tracking runtime | Camera capture, SCRFD detection, and ByteTrack tracking |
| Attendance runtime | Observation quality checks, PAD voting, ArcFace gallery matching, and event deduplication per session |
| Interactive demo | Preview, enrollment, identity matching, and PAD score monitoring |
| Evaluation tools | Component diagnostics, model inspection, and labeled liveness capture to CSV |

The interactive demo uses its own identity decision flow. Its PAD display is a
monitor, not an attendance gate. The separate attendance runtime implements PAD
gating but is not yet connected to the demo or backend storage.

## Architecture

```text
Camera frame
    │
    ▼
SCRFD — face boxes and five landmarks
    │
    ▼
ByteTrack — track association across frames
    │
    ▼
Observation quality checks
    │
    ▼
InsightFace Liveness Add-on — PAD score and per-track voting
    │ accepted
    ▼
ArcFace — embedding and gallery matching
    │ matched
    ▼
Attendance event — application integration boundary
```

[`FaceTrackingRuntime`](vision_runtime/pipeline/face_tracking_runtime.hpp)
returns the frame, detections, and tracks together, keeping image data and face
observations from the same capture.
[`BiometricAttendanceRuntime`](vision_runtime/pipeline/biometric_attendance_runtime.hpp)
composes the subsequent PAD and recognition stages. The application supplies
model paths, an identity gallery, and recognition thresholds.

Image preprocessing and model inference use the GPU path. Detection decoding,
tracking, landmark transform fitting, similarity calculations, and application
decisions use the CPU. GPU resource sharing avoids full-image CPU round trips
in the runtime image path; synchronization and model-output readback still occur.

## Requirements

- Windows x64.
- Visual Studio 2022 or Build Tools 2022 with **Desktop development with C++**,
  the MSVC toolchain, and a Windows SDK.
- CMake **3.21 or later**. The supplied preset uses Visual Studio 2022, x64.
- For capture and inference: a compatible webcam, GPU/driver support for the
  project's Direct3D 11/12 and DirectML path, and the required ONNX model files.

The native build uses Windows SDK libraries and the C++ standard library.
Python, OpenCV, CUDA, and a separate ONNX Runtime package are not required.
Building the project and running the default unit tests does not require a
webcam or model weights.

## Build and test

Run all commands from the **repository root** in PowerShell:

```powershell
cmake --preset windows-x64
cmake --build --preset release --parallel
ctest --preset test-release
```

Build output is written to `out/build/windows-x64/`. Executables are under
`tools/Release/`; test executables are under `tests/Release/` within that directory.

For a Debug build:

```powershell
cmake --build --preset debug --parallel
ctest --preset test-debug
```

## Model setup

Obtain the artifacts from the upstream sources recorded in
[`models/manifest.yaml`](models/manifest.yaml), place them at the paths below,
and verify their SHA-256 checksums against the manifest. Model binaries are
excluded from Git and must be provided separately.

| Path relative to `models/` | Role |
|---|---|
| `insightface/buffalo_sc/det_500m.onnx` | Default SCRFD face detector |
| `insightface/addons/liveness.onnx` | Selected PAD model |
| `insightface/buffalo_l/w600k_r50.onnx` | ArcFace identity embeddings |
| `insightface/buffalo_l/det_10g.onnx` | Optional alternative detector artifact |

The alternative detector is not required by the default pipeline.
`model_inspect --all` checks all four listed artifacts, including the alternative.
To inspect only one artifact, pass its path instead:

```powershell
Get-FileHash .\models\insightface\addons\liveness.onnx -Algorithm SHA256
.\out\build\windows-x64\tools\Release\model_inspect.exe models/insightface/addons/liveness.onnx
```

Model provenance and usage restrictions are recorded in the manifest. Model
weights have their own terms, distinct from the upstream implementation code.

### PAD configuration

The selected model is **InsightFace Liveness Add-on**, with the following native
input and decision contract:

- Five-landmark alignment to an **80 × 80** face crop.
- RGB float32 input in NCHW layout, with pixel values divided by 255.
- GPU preprocessing with replicated borders and rejection when more than 30%
  of the aligned crop falls outside the source image.
- A single `live_score`; the current development threshold is **0.95**, with
  equality accepted.

Rejected inputs carry no score and contribute neither a live nor a spoof vote.
The attendance runtime defaults to two live votes to proceed, or two spoof
votes to reject, within at most three scored observations per track.

The model checksum, preprocessing identifier, and threshold are defined in
[`model_contract.hpp`](vision_runtime/liveness/model_contract.hpp).
`liveness_capture` verifies the selected model's checksum before collection.
The threshold is a development operating point, not a validated guarantee
across subjects, cameras, or attack types. Recognition thresholds require
separate calibration for the target gallery.

## Run

Inspect the available hardware and launch the interactive demo:

```powershell
.\out\build\windows-x64\tools\Release\env_probe.exe
.\out\build\windows-x64\tools\Release\ta_demo.exe --name DEMO
```

The demo discovers model files under `models/`. Press **E** to repeat enrollment,
**P** to toggle PAD monitoring, **R** to reset tracking and identity state, and
**Esc** to exit. Enrollment is held in memory; console attendance messages are
demonstration events rather than persisted records.

### Capture labeled liveness observations

Check arguments, model checksums, and model descriptors without opening the
camera or writing a CSV:

```powershell
.\out\build\windows-x64\tools\Release\liveness_capture.exe --label real --scenario live --subject S01 --session preflight --validate-only
```

Collect a session with one visible face:

```powershell
.\out\build\windows-x64\tools\Release\liveness_capture.exe --label real --scenario live --subject S01 --session live_01 --split dev --samples 120
```

Labels are supplied by the operator: `real/live` for a person directly in front
of the camera; `spoof/photo`, `spoof/display`, or `spoof/replay` for the respective
attack scenarios. Use a unique session identifier for each run.

CSV output is written under `calibration/liveness/` relative to the working
directory. It records scores, decisions, input status, timing, and model
provenance; it does not save images. Existing output files are never overwritten.
The sample target counts successful measured observations, excluding warmup and
rejected inputs. Use `--help` for the complete option list.

## Tools

| Executable | Purpose |
|---|---|
| `env_probe` | Inspect Windows, GPU adapters, and camera availability |
| `model_inspect` | Inspect ONNX input/output descriptors and known provenance |
| `camera_probe` | Examine camera modes and capture performance |
| `scrfd_probe` | Evaluate face detection and timing |
| `bytetrack_probe` | Inspect tracking continuity and pipeline timing |
| `liveness_probe` | Observe PAD scores, rejected inputs, and timing |
| `liveness_capture` | Collect labeled PAD observations to CSV |
| `arcface_probe` | Examine enrollment and embedding stability |
| `ta_demo` | Preview tracking, enrollment, identity matching, and PAD scores |

## Validation

The default CTest suite contains three tests: `camera.format_selector`,
`camera.contract`, and `liveness.contract`. They cover camera configuration
policies, lifecycle contracts without capture, and PAD decision/voting and
capture metadata rules.

A separate hardware test checks GPU preprocessing on synthetic NV12 frames.
Passing a model path additionally exercises actual GPU inference and transitions
between rejected and accepted inputs:

```powershell
.\out\build\windows-x64\tests\Release\liveness_gpu_preprocess_test.exe models/insightface/addons/liveness.onnx
```

The current development-machine validation passed the Release build, all three
unit tests, and 13 synthetic GPU preprocessing cases, including actual model
inference. These checks establish implementation behavior on that machine;
they do not establish PAD accuracy or complete attendance-system correctness.

Broader subject and attack evaluation, attendance-state integration tests,
persistent gallery storage, and backend transport remain outstanding.

## Repository layout

```text
.
├── CMakeLists.txt           # Build entry point
├── CMakePresets.json        # Windows x64 configure, build, and test presets
├── vision_runtime/         # Camera, detection, tracking, PAD, recognition, orchestration
├── tools/                  # Diagnostic and interactive executables
├── tests/                  # Unit and hardware-dependent tests
├── models/                 # Artifact manifest and locally supplied weights
└── out/                    # Generated build output; excluded from Git
```

Local calibration data, model binaries, editor state, and internal learning
documentation are excluded from version control. The public build is configured
entirely through the files in this repository's root and source directories.
