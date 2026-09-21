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
background synchronization. Unit tests exercise failure handling with local
mocks; a separate native integration test connects these same storage and sync
components to the ASP.NET Core backend over HTTPS. Broader biometric evaluation
remains in progress.

| Component | Current implementation |
|---|---|
| Tracking runtime | Camera capture, SCRFD detection, and ByteTrack tracking |
| Attendance runtime | Landmark/quality checks, PAD voting, bounded recognition retry, gallery matching, and identity cooldown |
| Interactive demo | Supervised enrollment and preview using the shared PAD/recognition decision engine |
| Edge application | SQLite outbox/gallery, UUIDv7 events, DPAPI credentials, and background HTTP synchronization |
| Evaluation tools | Component diagnostics, model inspection, and labeled liveness capture to CSV |

`ta_demo` keeps its enrollment and console events in memory. `edge_client` stores
accepted events locally before upload and applies versioned galleries between
frames. Both use `AttendanceProcessor` for biometric decisions. See
[offline setup and recovery](#offline-first-application) below. Serialization and
parsing live in [`wire_format.cpp`](edge_app/sync/wire_format.cpp). The companion
[backend](https://github.com/MeinFinalProject/backend) implements these contracts
under `/api/v1`.

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
calibration ID or explicit development opt-in is required by the vision runtime.
Deployment configuration additionally requires a report reference and matching
report/model checksums; there is no default production recognition threshold.

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
An empty gallery produces no events. Import a compatible snapshot or use the
[backend development setup](#backend-development) below. Local import uses
`--import-gallery <snapshot.json>`; accepted input is defined by the existing
[`parse_gallery_file`](edge_app/sync/wire_format.cpp) implementation and its tests.
`--validate-only` checks configuration, model files/hash (including the selected PAD checksum), and stored gallery
metadata without starting camera/inference. Use Ctrl+C to stop capture.

Event UUIDs and occurrence times survive retries. Partial/missing receipts keep
unacknowledged rows pending; 401/403 pauses requests until credentials change.
The worker never accesses the vision runtime. Local COMMIT is required before
reporting an event as persisted.

If a database write or COMMIT fails, `PERSISTENCE_BLOCKED` reports the SQLite
error code and number of uncommitted events. The coordinator retains one frame's
batch in memory, including its original UUIDs, timestamps and gallery version.
The application retries persistence once per second and accepts no new biometric
decisions or gallery installations until that batch is saved. Camera acquisition
may continue filling its latest-frame mailbox. Successful rows are individually
reported as `COMMITTED`; `PERSISTENCE_RECOVERED` marks resumption.

Biometric recognition and its cooldown are not durable attendance confirmation.
Persistence retries do not reset cooldown or regenerate event IDs. If Ctrl+C
stops the process with pending records, it exits unsuccessfully with
`PERSISTENCE_INCOMPLETE`. A crash can lose records that exist only in RAM; they
require recapture. Previously committed events survive process restart and remain
eligible for upload. Repair capacity/permissions before restarting; do not delete
the database to clear an error. Physical power-loss guarantees depend on storage
hardware and are not established by the process-crash tests.

### Configuration reference

`edge_client --config <file>` loads the selected JSON file. For local backend
testing, edit `edge-config.backend.local.json`; the tracked
`edge-config.example.json` is a template. Paths resolve relative to the JSON
file's directory unless absolute. Restart Edge after editing its configuration.

| JSON fields | Responsibility |
| --- | --- |
| `schema_version`, `mode` | Configuration format and development/deployment validation |
| `device_id`, `base_url` | Registered device identity and versioned API address |
| `database`, `credential_file` | SQLite storage and the DPAPI-protected credential file |
| `scrfd_model`, `arcface_model`, `pad_model` | Local detection, recognition and liveness model files |
| `recognition_threshold`, `recognition_min_margin` | Identity-match acceptance and ambiguity threshold |
| `calibration_id`, `allow_uncalibrated_thresholds` | Calibration reference or explicit development opt-in |
| `allow_loopback_http` | Plain HTTP opt-in for loopback tests; keep false for backend HTTPS |

These are the currently supported operator settings. Additional runtime policies
are defined in C++ and are **not read from JSON**:

| Policy | Current setting and source |
| --- | --- |
| Camera | Index 0, requested 1280 x 720 at 30 FPS; actual format depends on the device. [`CameraConfig`](vision_runtime/camera/camera.hpp) |
| Detection/tracking | SCRFD candidate threshold 0.10 in [`load_config`](edge_app/app/application_config.cpp), with [`ByteTrackConfig`](vision_runtime/bytetrack/bytetrack.hpp) association defaults |
| Biometric quality | Detection score at least 0.50, face at least 64 x 64 pixels, maximum alignment RMSE 8. [`BiometricAttendanceConfig`](vision_runtime/pipeline/biometric_attendance_runtime.hpp) |
| PAD and recognition retries | PAD voting 2 of 3; recognition attempts/evidence policy and 30-second identity cooldown. [`BiometricAttendanceConfig`](vision_runtime/pipeline/biometric_attendance_runtime.hpp) |
| Gallery polling | 60 seconds after a successful request. [`GallerySync`](edge_app/sync/gallery_sync.cpp) |

Changing these C++ policies requires rebuilding. Adding an arbitrary JSON field
does not configure the corresponding runtime policy. PAD model checksums and
preprocessing are part of the model contract, rather than operator switches.
The deployment template additionally requires calibration evidence and checksums.

### Backend development

Keep the [backend](https://github.com/MeinFinalProject/backend) in a separate
checkout and follow its database and enrollment-worker setup. Start its HTTPS
launch profile, then run this command from the **backend checkout**:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/Initialize-EdgeDevelopment.ps1 -EdgeSource 'C:\path\to\edge-client'
```

Setup checks the shared ArcFace checksum, provisions a development device, and
creates the ignored `edge-config.backend.local.json` here. It uses
`https://localhost:7143/api/v1`, a Windows-user DPAPI credential in the backend's
private `.local` directory, and a separate SQLite database under
`data/backend-development/edge-dev-01/`. Repeated setup preserves the credential,
profile and outbox. Keep both checkouts at their configured locations and run
them as the same Windows user. HTTPS certificate validation remains enabled.

From the **Edge checkout**, validate the profile and start capture:

```powershell
.\out\build\windows-x64\tools\Release\edge_client.exe --config edge-config.backend.local.json --validate-only
.\out\build\windows-x64\tools\Release\edge_client.exe --config edge-config.backend.local.json
```

Validation starts no camera and sends no network request. Capture uses the
existing `edge_client` runtime, including PAD, gallery polling and outbox retries.
Run one Edge process per database; Ctrl+C stops it. Once a gallery is cached,
capture can continue while the backend is unavailable. Do not delete the database
to simulate offline operation: stop the backend, then restart it to let pending
events drain. Gallery polling normally runs every 60 seconds.

For everyday use, run the executable directly; the setup helper is not a runtime
dependency. Enrollment photos can be selected directly in the backend's Swagger
UI using `POST /api/v1/biometric-enrollments/{id}/samples/upload`. Follow the
backend README's **Manual walkthrough with Swagger** for enrollment, approval,
gallery synchronization and academic attendance checks.

For an optional automated test without camera capture, run from the **backend checkout**:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/Test-EdgeIntegration.ps1 -EdgeSource 'C:\path\to\edge-client'
```

This links the actual Edge HTTP, wire-format, SQLite and synchronization source
into a native probe. It checks authentication, gallery download/persistence,
ETag `304`, offline persistence, retry state after database reopening, and
duplicate delivery following a lost acknowledgement. Successful requests reach
the running ASP.NET Core API and PostgreSQL; connection failure and acknowledgement
loss are injected by the probe. Its temporary database is separate from the
capture database. Each run stores one synthetic `backend-contract-probe` raw event
in the development backend; it does not enroll a person or award attendance.

Real recognition requires approved biometric enrollment and a synchronized
nonempty gallery. Academic attendance additionally requires approved KRS, an
actual teaching session, and the device's classroom assignment established before
the observation. Inspect the backend event decision and session roster: an
`accepted` receipt alone means the raw event was stored. The generated profile is
explicitly for development, with uncalibrated recognition thresholds.

### Deployment configuration

[`edge-config.example.json`](edge-config.example.json) preserves research mode.
[`edge-config.deployment.example.json`](edge-config.deployment.example.json) is an
intentionally incomplete deployment template: it must fail validation unchanged.
No deployment biometric thresholds or calibration results are supplied.

```powershell
Copy-Item edge-config.deployment.example.json edge-config.deployment.local.json
# Fill the local copy using the reviewed calibration results, then validate:
.\out\build\windows-x64\tools\Release\edge_client.exe --config edge-config.deployment.local.json --require-deployment --validate-only
```

Deployment requires `mode: "deployment"`, a device ID, calibrated numeric
recognition threshold/margin, a nonempty calibration ID, a readable nonempty
`calibration_report`, its lowercase SHA-256 in `calibration_report_sha256`, and
the calibrated ArcFace model SHA-256 in `calibration_model_sha256`. It forbids
`allow_uncalibrated_thresholds` and `allow_loopback_http`. `--require-deployment`
also rejects accidentally passing the development config. An empty `base_url`
still allows offline operation while the backend is unavailable.

The report checksum and model binding establish which evidence was supplied;
they cannot prove its scientific validity. A reviewer must verify the chosen
thresholds/margin against that report, dataset split, evaluation and target
conditions. An arbitrary calibration ID or a renamed file is not calibration.
Changing model, operating point or experiment requires reviewing the evidence
and local config together. `--validate-only` opens/migrates storage and applies
file permissions, but starts no camera or inference and contacts no backend.

### Local data protection

Use a dedicated local directory on a filesystem with Windows ACL support.
The examples use `data/`; a dedicated `%LOCALAPPDATA%\EdgeClient\data` directory
under the research device account is preferable for operation outside a checkout.
Set absolute paths in the local config when using it; environment-variable
placeholders inside JSON are not expanded. Do not point data paths at a shared
folder, profile root, checkout root, network share, junction or symbolic link.

Startup/provisioning restricts the data directory and database/credential files
to the current Windows user and SYSTEM, with inheritance protecting SQLite
journals and temporary files. Existing credential ACLs are checked/applied at
application startup. Permission failure stops initialization visibly; file paths
with reparse points or multiple hard links are rejected. Use a directory devoted
to Edge data because setting its inherited ACL can affect its other children.

DPAPI protects credential contents for the Windows user. Provision through the
hidden interactive prompt, never a token argument or a committed JSON file:

```powershell
.\out\build\windows-x64\tools\Release\edge_client.exe --provision-token data/deployment/device.dpapi
```

The database itself remains ordinary SQLite and contains biometric templates.
**Enable and verify BitLocker/device encryption on volumes containing operational
data and backups before collecting deployment data.** ACLs do not protect a disk
removed from the device; disk encryption addresses that offline access threat.
Follow the device's recovery-key policy. These changes do not enable encryption
or claim to protect against an administrator, SYSTEM, or malware running as the
same user. See [Microsoft's BitLocker overview](https://learn.microsoft.com/en-us/windows/security/operating-system-security/data-protection/bitlocker/)
and [Windows object security descriptors](https://learn.microsoft.com/en-us/windows/win32/secauthz/creating-a-security-descriptor-for-a-new-object-in-c--).

Production application logs contain operational status, opaque event UUIDs and
error codes, not tokens, embedding vectors or event payloads. Research probes
can export biometric observations: keep their CSV/logs and calibration reports
inside ignored private directories. `.gitignore` excludes `data/`, calibration
outputs, SQLite files/sidecars, DPAPI blobs and local configuration copies; this
does not encrypt files or remove anything already tracked in Git.

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

The default CTest suite contains fourteen tests: camera configuration/lifecycle, PAD
contracts, shared attendance decisions, durable storage/synchronization, and
native HTTP against a loopback mock, wire format, credentials, and coordinator
gallery/commit boundaries. Tests cover 100 offline events, forced
process termination around commits, gallery rollback, lost/partial/invalid ACKs,
authentication blocking, retry delays, DPAPI, redirects, timeouts, and response
size limits. Additional regressions cover database v1-to-v2 migration, endpoint
backoff across new events/restarts, 10,000-template replacement/rollback, HTTP
error bodies, and inconclusive zero embedding aggregates. No production backend,
webcam, or weights are required for these tests.

Hardening coverage includes real SQLite write denial, a rejected COMMIT,
`SQLITE_FULL` via a test-only page cap, recovery with stable IDs, partial frame
batch failure, and receipt handling when the local acknowledgement COMMIT fails.
ACL tests inspect the database, live rollback journal and rotated credential.
Deployment tests reject dev opt-in, missing evidence and checksum mismatch.
Communication tests include incomplete/corrupt galleries, non-finite embeddings,
duplicate/partial/malformed receipts and retryable HTTP errors. These tests check
Edge-side behavior only. The separate backend integration command above checks
the live ASP.NET Core contract and local recovery behavior.

A separate hardware test checks GPU preprocessing on synthetic NV12 frames.
Passing a model path additionally exercises actual GPU inference and transitions
between rejected and accepted inputs:

```powershell
.\out\build\windows-x64\tests\Release\liveness_gpu_preprocess_test.exe models/insightface/addons/liveness.onnx
```

Unit, integration and GPU checks establish implementation behavior on the test
machine; they do not establish PAD accuracy or complete attendance-system
correctness under field conditions.

Broader subject/attack evaluation, calibrated recognition thresholds, real
walk-through time-to-verified measurements, and deployment validation
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
├── edge-config.example.json # Offline development configuration
├── edge-config.deployment.example.json # Deployment template; requires calibration evidence
├── tools/                  # Application, diagnostic and interactive executables
├── tests/                  # Unit and hardware-dependent tests
├── models/                 # Artifact manifest and locally supplied weights
└── out/                    # Generated build output; excluded from Git
```

Local calibration data, model binaries, editor state, and internal learning
documentation are excluded from version control. The public build is configured
entirely through the files in this repository's root and source directories.
