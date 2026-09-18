# Edge Client

Native Windows computer vision client for a biometric attendance research project.
Built with **C++20**, **Windows ML / DirectML**, and **Direct3D 11/12**, it provides
camera capture, face detection and tracking, presentation attack detection (PAD),
and face recognition.

This repository contains the edge runtime, diagnostic tools, and tests. The
frontend and backend are maintained in separate repositories.

## Project status

The project is an active research prototype. Individual vision components and
attendance orchestration now share a decision engine with the interactive demo.
A native application adds local event durability, gallery persistence, and
background synchronization. The backend API remains a proposal verified with
local mocks; broader biometric evaluation remains in progress.

| Component | Current implementation |
|---|---|
| Tracking runtime | Camera capture, SCRFD detection, and ByteTrack tracking |
| Attendance runtime | Landmark/quality checks, PAD voting, bounded recognition retry, gallery matching, and identity cooldown |
| Interactive demo | Supervised enrollment and preview using the shared PAD/recognition decision engine |
| Edge application | SQLite outbox/gallery, UUIDv7 events, DPAPI credentials, and background HTTP synchronization |
| Evaluation tools | Component diagnostics, model inspection, and labeled liveness capture to CSV |

`ta_demo` keeps its enrollment and console events in memory. `edge_client` stores
accepted events locally before upload and applies versioned galleries between
frames. Both use `AttendanceProcessor` for biometric decisions. The API and
offline setup are documented in [the edge v1 contract](contracts/edge-v1.md).

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
Attendance event → SQLite outbox → HTTP worker → per-event receipt
```

[`FaceTrackingRuntime`](vision_runtime/pipeline/face_tracking_runtime.hpp)
returns the frame, detections, and tracks together, keeping image data and face
observations from the same capture.
[`BiometricAttendanceRuntime`](vision_runtime/pipeline/biometric_attendance_runtime.hpp)
composes the subsequent PAD and recognition stages. The application supplies
model paths, an identity gallery, and explicit recognition thresholds. Empty
galleries are allowed at startup but cannot produce attendance events. A
calibration ID or explicit development opt-in is required; there is no default
production recognition threshold.

Image preprocessing and model inference use the GPU path. Detection decoding,
tracking, landmark transform fitting, similarity calculations, and application
decisions use the CPU. GPU resource sharing avoids full-image CPU round trips
in the runtime image path; synchronization and model-output readback still occur.

The application code is organized by responsibility:

```text
edge_app/
  domain/    Attendance records, gallery snapshots, UUID generation
  storage/   Shared Database, schema, outbox and gallery repositories
  sync/      HTTP/wire format, upload, gallery sync, retry, worker and network hints
  security/  Device credentials and model checksums
  app/       Configuration, attendance coordinator and application lifecycle
```

`tools/edge_client.cpp` handles CLI and console controls. `EdgeApplication`
owns the database and repositories and manages capture/synchronization lifetime.
`AttendanceCoordinator` commits accepted events and installs gallery snapshots
on the vision owner thread. Repositories share one database connection; both
sync tasks share one HTTP client and one worker. Domain headers contain only
standard C++ data types. These folders still build into one `edge_app_core` library.

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
**P** to toggle the PAD display, **R** to reset tracking and identity state, and
**Esc** to exit. Enrollment is held in memory; console attendance messages are
demonstration events rather than persisted records. PAD remains mandatory for
demo recognition; `--no-pad` is an enrollment/preview diagnostic mode only.

### Offline-first application

```powershell
Copy-Item edge-config.example.json edge-config.local.json
.\out\build\windows-x64\tools\Release\edge_client.exe --config edge-config.local.json --validate-only
.\out\build\windows-x64\tools\Release\edge_client.exe --config edge-config.local.json
```

The example disables networking and explicitly uses a development threshold.
An empty gallery produces no events. Import a compatible snapshot or configure
the future server endpoint as described in the [API contract](contracts/edge-v1.md).
`--validate-only` checks configuration, model files/hash (including the selected PAD checksum), and stored gallery
metadata without starting camera/inference. Use Ctrl+C to stop capture.

Event UUIDs and occurrence times survive retries. Partial/missing receipts keep
unacknowledged rows pending; 401/403 pauses requests until credentials change.
The worker never accesses the vision runtime. Local COMMIT is required before
reporting an event as persisted.

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
| `ta_demo` | Preview/enrollment with shared PAD and recognition decisions |
| `edge_client` | Offline-first capture, durable outbox/gallery, credential provisioning, and background sync |
| `biometric_benchmark` | Synthetic GPU 1/2/3-crop service times and CPU gallery benchmark to CSV |

## Validation

The default CTest suite contains eleven tests: camera configuration/lifecycle, PAD
contracts, shared attendance decisions, durable storage/synchronization, and
native HTTP against a loopback mock, wire format, credentials, and coordinator
gallery/commit boundaries. Tests cover 100 offline events, forced
process termination around commits, gallery rollback, lost/partial/invalid ACKs,
authentication blocking, retry delays, DPAPI, redirects, timeouts, and response
size limits. Additional regressions cover database v1-to-v2 migration, endpoint
backoff across new events/restarts, 10,000-template replacement/rollback, HTTP
error bodies, and inconclusive zero embedding aggregates. No production backend,
webcam, or weights are required for these tests.

A separate hardware test checks GPU preprocessing on synthetic NV12 frames.
Passing a model path additionally exercises actual GPU inference and transitions
between rejected and accepted inputs:

```powershell
.\out\build\windows-x64\tests\Release\liveness_gpu_preprocess_test.exe models/insightface/addons/liveness.onnx
```

The current development-machine validation passed the Release build, all eleven
CTest cases, and 13 synthetic GPU preprocessing cases, including actual model
inference. These checks establish implementation behavior on that machine;
they do not establish PAD accuracy or complete attendance-system correctness.

Broader subject/attack evaluation, calibrated recognition thresholds, real
walk-through time-to-verified measurements, and production backend integration
remain outstanding. Synthetic benchmarks measure service time, not biometric
accuracy. `evaluate_many` still evaluates faces serially; true batching and GPU
gallery matching remain measurement-driven optimizations.

```powershell
.\out\build\windows-x64\tools\Release\biometric_benchmark.exe out/benchmarks/session-01.csv models
```

Use a new output filename for each run. The tool does not open a camera.

## Repository layout

```text
.
├── CMakeLists.txt           # Build entry point
├── CMakePresets.json        # Windows x64 configure, build, and test presets
├── vision_runtime/         # Camera, detection, tracking, PAD, recognition, orchestration
├── edge_app/               # Local storage, credentials, event contract, HTTP and worker
├── contracts/              # Proposed API contract for the future backend
├── edge-config.example.json # Offline development configuration
├── tools/                  # Application, diagnostic and interactive executables
├── tests/                  # Unit and hardware-dependent tests
├── models/                 # Artifact manifest and locally supplied weights
└── out/                    # Generated build output; excluded from Git
```

Local calibration data, model binaries, editor state, and internal learning
documentation are excluded from version control. The public build is configured
entirely through the files in this repository's root and source directories.
