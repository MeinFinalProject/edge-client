// Service-time benchmark using a synthetic GPU image, not a human accuracy test.
#include "arcface/arcface.hpp"
#include "liveness/liveness.hpp"
#include "pipeline/gallery_matcher.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

using namespace vision_runtime;
namespace
{
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point begin)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}
double percentile(std::vector<double> values, double fraction)
{
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(std::ceil(values.size() * fraction)) - 1];
}
void row(std::ostream &out, char const *stage, int count, std::vector<double> const &values)
{
    out << "synthetic," << stage << ',' << count << ',' << values.size() << ',' << percentile(values, .5) << ','
        << percentile(values, .95) << '\n';
}
} // namespace
int main(int argc, char **argv)
{
    try
    {
        if (argc < 2 || argc > 3)
        {
            std::cout << "biometric_benchmark <new-output.csv> [models-directory]\nSynthetic GPU PAD/ArcFace 1/2/3 "
                         "crops and CPU gallery service time; no camera.\n";
            return argc == 1 ? 0 : 1;
        }
        std::filesystem::path output = argv[1], models = argc == 3 ? argv[2] : "models";
        if (std::filesystem::exists(output))
            throw std::runtime_error("Output exists; choose a new file");
        if (output.has_parent_path())
            std::filesystem::create_directories(output.parent_path());
        std::ofstream csv(output);
        if (!csv)
            throw std::runtime_error("Cannot create output");
        csv << "condition,stage,item_count,iterations,median_ms,p95_ms\n";
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
        winrt::com_ptr<ID3D11Device> device;
        winrt::com_ptr<ID3D11DeviceContext> context;
        winrt::check_hresult(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                               D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION,
                                               device.put(), nullptr, context.put()));
        constexpr int width = 640, height = 480;
        std::vector<unsigned char> pixels(width * height * 3 / 2, 128);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                pixels[y * width + x] = static_cast<unsigned char>(16 + (x * 13 + y * 7) % 200);
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_NV12;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA data{pixels.data(), width, 0};
        GpuFrame frame;
        winrt::check_hresult(device->CreateTexture2D(&desc, &data, frame.texture.put()));
        frame.width = width;
        frame.height = height;
        frame.dxgi_format = DXGI_FORMAT_NV12;
        frame.frame_id = 1;
        FaceDetection face;
        face.x1 = 100;
        face.y1 = 100;
        face.x2 = 324;
        face.y2 = 324;
        face.score = .99F;
        face.landmarks = {Point2f{38.2946F, 51.6963F}, Point2f{73.5318F, 51.5014F}, Point2f{56.0252F, 71.7366F},
                          Point2f{41.5493F, 92.3655F}, Point2f{70.7299F, 92.2041F}};
        for (auto &p : face.landmarks)
        {
            p.x = p.x * 2 + 100;
            p.y = p.y * 2 + 100;
        }
        liveness::LivenessConfig pad_config;
        pad_config.model_path = models / "insightface/addons/liveness.onnx";
        arcface::ArcFaceConfig arc_config;
        arc_config.model_path = models / "insightface/buffalo_l/w600k_r50.onnx";
        liveness::LivenessDetector pad(pad_config);
        arcface::ArcFaceRecognizer arc(arc_config);
        for (int count : {1, 2, 3})
        {
            std::vector<FaceDetection> faces(count, face);
            std::vector<double> pad_times, arc_times, total_times;
            for (int iteration = -3; iteration < 20; ++iteration)
            {
                auto begin = Clock::now();
                auto live = pad.evaluate_many(frame, faces);
                double pad_ms = ms(begin);
                for (auto const &r : live)
                    if (!r.live_score)
                        throw std::runtime_error("Synthetic crop rejected; benchmark invalid");
                begin = Clock::now();
                auto embeddings = arc.evaluate_many(frame, faces);
                double arc_ms = ms(begin);
                if (embeddings.size() != faces.size())
                    throw std::runtime_error("Missing embedding");
                if (iteration >= 0)
                {
                    pad_times.push_back(pad_ms);
                    arc_times.push_back(arc_ms);
                    total_times.push_back(pad_ms + arc_ms);
                }
            }
            row(csv, "pad_serial", count, pad_times);
            row(csv, "arcface_serial", count, arc_times);
            row(csv, "pad_plus_arcface_serial", count, total_times);
            std::cout << count << " crops: PAD p95=" << percentile(pad_times, .95)
                      << " ms; ArcFace p95=" << percentile(arc_times, .95) << " ms\n";
        }
        std::mt19937 random(17);
        std::normal_distribution<float> distribution;
        for (int count : {1, 100, 1000, 10000})
        {
            std::vector<pipeline::GalleryTemplate> templates(count);
            for (int i = 0; i < count; ++i)
            {
                templates[i].identity_id = std::to_string(i);
                for (auto &v : templates[i].embedding)
                    v = distribution(random);
            }
            pipeline::GalleryMatcher matcher;
            matcher.replace(templates);
            auto query = templates[0].embedding;
            std::vector<double> times;
            for (int i = -3; i < 100; ++i)
            {
                auto start = Clock::now();
                auto match = matcher.match(query);
                double elapsed = ms(start);
                if (match.identity_id != "0")
                    throw std::runtime_error("Gallery result mismatch");
                if (i >= 0)
                    times.push_back(elapsed);
            }
            row(csv, "gallery_cpu", count, times);
            std::cout << count << " gallery templates: CPU p95=" << percentile(times, .95) << " ms\n";
        }
        csv.flush();
        if (!csv)
            throw std::runtime_error("Benchmark write failed");
        std::cout << "No SCRFD/camera/voting in timing; this does not measure time-to-verified or accuracy.\n";
        return 0;
    }
    catch (winrt::hresult_error const &e)
    {
        std::cerr << "Windows/GPU failure " << std::hex << e.code().value << '\n';
        return 1;
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
