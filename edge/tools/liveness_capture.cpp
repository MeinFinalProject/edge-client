#include "capture_support.hpp"
#include "liveness/liveness.hpp"
#include "pipeline/face_tracking_runtime.hpp"

#include <Windows.h>
#include <bcrypt.h>
#include <winrt/base.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
namespace fs = std::filesystem;
using capture::number;
using Clock = std::chrono::steady_clock;

std::string utc_stamp() {
    const auto now = std::time(nullptr);
    std::tm tm{};
    if (gmtime_s(&tm, &now)) throw std::runtime_error("Cannot read UTC time");
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%dT%H%M%SZ");
    return out.str();
}

fs::path find_model(fs::path root, fs::path const& relative) {
    while (!root.empty()) {
        for (auto const& prefix : {fs::path{}, fs::path{"edge"}}) {
            auto candidate = root / prefix / "models" / relative;
            if (fs::is_regular_file(candidate)) return fs::absolute(candidate);
        }
        auto parent = root.parent_path();
        if (parent == root) break;
        root = parent;
    }
    return {};
}

fs::path model_path(fs::path path, fs::path const& relative) {
    if (path.empty()) path = find_model(fs::current_path(), relative);
    if (path.empty()) {
        std::wstring exe(32768, L'\0');
        const auto n = GetModuleFileNameW(nullptr, exe.data(), static_cast<DWORD>(exe.size()));
        if (n == 0 || n >= exe.size()) throw std::runtime_error("Cannot locate executable");
        exe.resize(n);
        path = find_model(fs::path{exe}.parent_path(), relative);
    }
    if (!fs::is_regular_file(path)) throw std::runtime_error("Model not found: " + relative.string());
    return fs::absolute(path).lexically_normal();
}

// Hash the actual loaded artifact, including user-supplied paths. Do not copy
// a hardcoded manifest checksum into the CSV and pretend it was measured.
std::string sha256(fs::path const& path) {
    struct Hash {
        BCRYPT_ALG_HANDLE algorithm = nullptr;
        BCRYPT_HASH_HANDLE hash = nullptr;
        ~Hash() {
            if (hash) BCryptDestroyHash(hash);
            if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
        }
    } state;
    auto check = [](NTSTATUS result) {
        if (result < 0) throw std::runtime_error("BCrypt SHA-256 failed");
    };
    check(BCryptOpenAlgorithmProvider(&state.algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0));
    check(BCryptCreateHash(state.algorithm, &state.hash, nullptr, 0, nullptr, 0, 0));
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read model for SHA-256");
    std::array<unsigned char, 65536> buffer{};
    while (file.read(reinterpret_cast<char*>(buffer.data()), buffer.size()) || file.gcount())
        check(BCryptHashData(state.hash, buffer.data(), static_cast<ULONG>(file.gcount()), 0));
    if (!file.eof()) throw std::runtime_error("Model read failed during SHA-256");
    std::array<unsigned char, 32> digest{};
    check(BCryptFinishHash(state.hash, digest.data(), static_cast<ULONG>(digest.size()), 0));
    std::ostringstream out;
    for (auto byte : digest) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return out.str();
}

class Csv {
    HANDLE handle_ = INVALID_HANDLE_VALUE;
public:
    explicit Csv(fs::path const& path) {
        fs::create_directories(path.parent_path());
        handle_ = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                              CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Cannot create CSV (existing files are never overwritten): " + path.string());
    }
    ~Csv() { if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_); }
    Csv(Csv const&) = delete;
    Csv& operator=(Csv const&) = delete;
    void write(std::vector<std::string> const& fields) {
        const auto line = capture::csv_line(fields);
        DWORD written = 0;
        if (!WriteFile(handle_, line.data(), static_cast<DWORD>(line.size()), &written, nullptr) ||
            written != line.size() || !FlushFileBuffers(handle_))
            throw std::runtime_error("CSV write/flush failed; session incomplete");
    }
};

struct Args {
    fs::path scrfd, addons, output;
    std::string label, scenario, subject, session, split = "dev";
    int samples = 120, every = 5, max_frames = 5000, warmup = 5;
    float min_score = .5F;
    bool validate_only = false;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--validate-only") { a.validate_only = true; continue; }
        if (key == "--help" || key == "-h") {
            std::cout <<
                "liveness_capture: Selected InsightFace addon, one observation per CSV row.\n"
                "Required: --label real|spoof --scenario live|photo|display|replay --subject S01\n"
                "  real/live = person; spoof/photo = print; spoof/display = still on screen;\n"
                "  spoof/replay = video on screen. Labels are supplied by the operator.\n"
                "Options: --split dev|test --session ID --output path.csv\n"
                "  --samples 120      successful measured observations (not camera frames)\n"
                "  --every 5          attempt every N processed tracking frames (not fixed FPS)\n"
                "  --warmup 5         successful model observations excluded from measured samples\n"
                "  --max-frames 5000  limit includes warmup; timeout also aborts\n"
                "  --min-det-score 0.5\n"
                "  --scrfd path.onnx --addons path.onnx\n"
                "  --validate-only    check arguments, model hashes/loading; no camera or CSV\n"
                "Default output: calibration/liveness/<split>_<subject>_<scenario>_<session>.csv\n"
                "One visible face only. TA live threshold 0.95. No image files saved.\n"
                "Exit: 0 target reached; 2 insufficient observations; 1 error (partial CSV retained).\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[++i];
        auto integer = [&] {
            std::size_t used = 0; const int n = std::stoi(value, &used);
            if (used != value.size() || n < 0 || n > 1000000)
                throw std::invalid_argument("Invalid integer for " + key);
            return n;
        };
        auto decimal = [&] {
            std::size_t used = 0; const float n = std::stof(value, &used);
            if (used != value.size() || !std::isfinite(n))
                throw std::invalid_argument("Invalid number for " + key);
            return n;
        };
        if (key == "--label") a.label = value;
        else if (key == "--scenario") a.scenario = value;
        else if (key == "--subject") a.subject = value;
        else if (key == "--session") a.session = value;
        else if (key == "--split") a.split = value;
        else if (key == "--output") a.output = fs::u8path(value);
        else if (key == "--scrfd") a.scrfd = fs::u8path(value);
        else if (key == "--addons") a.addons = fs::u8path(value);
        else if (key == "--samples") a.samples = integer();
        else if (key == "--every") a.every = integer();
        else if (key == "--warmup") a.warmup = integer();
        else if (key == "--max-frames") a.max_frames = integer();
        else if (key == "--min-det-score") a.min_score = decimal();
        else throw std::invalid_argument("Unknown argument: " + key);
    }
    capture::validate_label(a.label, a.scenario);
    if (a.session.empty()) a.session = utc_stamp() + "_" + number(GetCurrentProcessId());
    if (!capture::identifier(a.subject) || !capture::identifier(a.session))
        throw std::invalid_argument("subject/session require A-Z, a-z, 0-9, underscore or hyphen");
    if (a.split != "dev" && a.split != "test") throw std::invalid_argument("split must be dev or test");
    if (a.samples < 1 || a.every < 1 || a.max_frames < 1 || a.min_score < 0 || a.min_score > 1)
        throw std::invalid_argument("samples/every/max-frames >= 1; min-det-score 0..1");
    if (a.output.empty()) a.output = fs::path{"calibration/liveness"} /
        (a.split + "_" + a.subject + "_" + a.scenario + "_" + a.session + ".csv");
    a.output = fs::absolute(a.output);
    // Reject before loading models or starting the camera.
    if (fs::exists(a.output)) throw std::invalid_argument("Output already exists; choose a new session/output");
    a.scrfd = model_path(a.scrfd, "insightface/buffalo_sc/det_500m.onnx");
    a.addons = model_path(a.addons, vision_runtime::liveness::kModelRelativePath);
    return a;
}

std::string utf8(fs::path const& path) { return winrt::to_string(path.wstring()); }

} // namespace

int main(int argc, char** argv) {
    try {
        const auto a = parse(argc, argv);
        const auto addons_hash = sha256(a.addons), scrfd_hash = sha256(a.scrfd);
        if (addons_hash != vision_runtime::liveness::kModelSha256)
            throw std::runtime_error("Selected addon SHA-256 mismatch; update the TA selection explicitly");
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
        vision_runtime::pipeline::FaceTrackingConfig config;
        config.camera.width = 1280; config.camera.height = 720; config.camera.fps = 30;
        config.camera.source_subtype = L"NV12"; config.camera.output_subtype = L"NV12";
        config.scrfd.model_path = a.scrfd;
        config.scrfd.score_threshold = .1F; config.scrfd.nms_threshold = .4F;
        vision_runtime::pipeline::FaceTrackingRuntime runtime{config};
        vision_runtime::liveness::LivenessConfig addons_config;
        addons_config.model_path = a.addons;
        vision_runtime::liveness::LivenessDetector addons{addons_config};
        if (a.validate_only) {
            std::cout << "Model descriptors loaded (no inference). SHA-256:\naddons " << addons_hash
                      << "\nscrfd " << scrfd_hash << '\n';
            return 0;
        }
        Csv csv{a.output};
        const std::vector<std::string> header = {
            "schema_version","split","subject","session","scenario","label","phase","status",
            "utc_logged","camera_frame_id","app_timestamp_ns","camera_width","camera_height","track_id",
            "detection_score","face_x1","face_y1","face_x2","face_y2","live_score","decision","threshold",
            "out_of_bounds_ratio","wall_ms","error","model_sha256","scrfd_sha256","model_path","preprocess_contract",
            "every","warmup_observations","target_observations"
        };
        csv.write(header);
        std::cout << "CSV: " << a.output.string() << "\nLabel: " << a.label << " / " << a.scenario
                  << "\nKeep ONE face/presentation visible. Selected model: InsightFace liveness addon.\n"
                  << "Preparing " << a.warmup << " warmup observations, then " << a.samples << " measured observations...\n";
        runtime.start();
        int warm = 0, measured = 0, skipped = 0;
        std::uint64_t previous_id = 0;
        const auto start = Clock::now();
        for (int index = 1; index <= a.max_frames && measured < a.samples; ++index) {
            const auto frame = runtime.wait_for_frame(std::chrono::milliseconds{1500});
            if (!frame) throw std::runtime_error("Camera/tracking timeout; partial CSV retained");
            if (frame->gpu_frame.frame_id <= previous_id)
                throw std::runtime_error("Camera frame ID did not advance; refusing duplicate observations");
            previous_id = frame->gpu_frame.frame_id;
            if ((index - 1) % a.every) continue;

            const bool warming = warm < a.warmup;
            std::string status = "ok";
            const vision_runtime::bytetrack::TrackResult* track = nullptr;
            const auto faces = std::count_if(frame->detections.begin(), frame->detections.end(),
                [&](auto const& d) { return d.score >= a.min_score; });
            if (faces == 0) status = "no_face";
            else if (faces > 1 || frame->tracks.size() > 1) status = "multiple_faces";
            else if (frame->tracks.empty() || !frame->tracks.front().has_observation ||
                     frame->tracks.front().observation.score < a.min_score) status = "no_current_track";
            else track = &frame->tracks.front();

            vision_runtime::liveness::LivenessResult result;
            std::string error;
            double wall_ms = 0;
            if (track) {
                const auto begin = Clock::now();
                try {
                    result = addons.evaluate(frame->gpu_frame, track->observation);
                    if (!result.live_score) { status = "input_rejected"; ++skipped; }
                } catch (winrt::hresult_error const& e) { error = winrt::to_string(e.message()); }
                  catch (std::exception const& e) { error = e.what(); }
                wall_ms = std::chrono::duration<double, std::milli>(Clock::now()-begin).count();
                if (!error.empty()) status = "model_error";
            } else ++skipped;
            auto const& gpu = frame->gpu_frame;
            std::vector<std::string> row = {
                "liveness_capture_v1",a.split,a.subject,a.session,a.scenario,a.label,warming ? "warmup" : "measure",status,
                utc_stamp(),number(gpu.frame_id),number(gpu.timestamp_ns),number(gpu.width),number(gpu.height),
                track ? number(track->track_id) : "", track ? number(track->observation.score) : "",
                track ? number(track->observation.x1) : "",track ? number(track->observation.y1) : "",
                track ? number(track->observation.x2) : "",track ? number(track->observation.y2) : "",
                result.live_score ? number(*result.live_score) : "",
                track && error.empty() ? vision_runtime::liveness::to_string(result.decision) : "",
                number(vision_runtime::liveness::kTaLiveThreshold),
                track && error.empty() ? number(result.out_of_bounds_ratio) : "",track ? number(wall_ms) : "",error,
                addons_hash,scrfd_hash,utf8(a.addons),vision_runtime::liveness::kPreprocessContract,
                number(a.every),number(a.warmup),number(a.samples)
            };
            if (row.size() != header.size()) throw std::logic_error("CSV schema mismatch");
            csv.write(row);
            if (status == "model_error")
                throw std::runtime_error("Model failed; error recorded in CSV. Fix before continuing collection.");
            if (status != "ok") {
                if (skipped == 1 || skipped % 20 == 0) std::cout << "Skipped: " << status << " (" << skipped << ")\n";
                continue;
            }
            if (warming) ++warm; else ++measured;
            std::cout << (warming ? "warmup " : "observation ") << (warming ? warm : measured)
                      << " frame=" << gpu.frame_id << " label=" << a.label
                      << " real_score=" << *result.live_score << " " << vision_runtime::liveness::to_string(result.decision) << '\n';
        }
        runtime.stop();
        const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
        std::cout << "Measured observations: " << measured << '/' << a.samples << "; skipped attempts: " << skipped
                  << "; elapsed seconds (including warmup): " << seconds << '\n';
        return measured == a.samples ? 0 : 2;
    } catch (winrt::hresult_error const& e) {
        std::cerr << "liveness_capture: " << winrt::to_string(e.message()) << '\n';
    } catch (std::exception const& e) {
        std::cerr << "liveness_capture: " << e.what() << '\n';
    }
    return 1;
}
