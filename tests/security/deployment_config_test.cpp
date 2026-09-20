#include "app/application_config.hpp"
#include "security/model_hash.hpp"
#include "support/edge_fixtures.hpp"
using namespace edge_test;
int main()
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        auto root = std::filesystem::current_path() / "edge-test-data" / uuid_v7(utc_ms());
        std::filesystem::create_directories(root);
        auto file = root / "config.json", report = root / "synthetic-evidence.txt";
        {
            std::ofstream out(report);
            out << "TEST FIXTURE ONLY - not biometric calibration evidence";
        }
        JsonObject json = JsonObject::Parse(LR"({"schema_version":1,"mode":"development","device_id":"test",
            "base_url":"","database":"data/edge.db","credential_file":"data/token.dpapi",
            "scrfd_model":"det.onnx","arcface_model":"arc.onnx","pad_model":"pad.onnx",
            "recognition_threshold":0.75,"recognition_min_margin":0,"calibration_id":"",
            "allow_uncalibrated_thresholds":true,"allow_loopback_http":false})");
        auto load = [&] {
            {
                std::ofstream out(file);
                out << winrt::to_string(json.Stringify());
            }
            return load_config(file);
        };
        check(!load().deployment, "Development mode no longer works");
        json.SetNamedValue(L"mode", JsonValue::CreateNumberValue(1));
        rejects(load);
        json.SetNamedValue(L"mode", JsonValue::CreateStringValue(L"development"));
        json.SetNamedValue(L"allow_uncalibrated_thresholds", JsonValue::CreateStringValue(L"true"));
        rejects(load);
        json.SetNamedValue(L"allow_uncalibrated_thresholds", JsonValue::CreateBooleanValue(true));
        json.SetNamedValue(L"mode", JsonValue::CreateStringValue(L"deployment"));
        rejects(load); // Cannot smuggle the dev opt-in into deployment.
        json.SetNamedValue(L"allow_uncalibrated_thresholds", JsonValue::CreateBooleanValue(false));
        rejects(load); // No calibration identifier/evidence.
        json.SetNamedValue(L"calibration_id", JsonValue::CreateStringValue(L"arbitrary-id-is-not-evidence"));
        rejects(load);
        json.Insert(L"calibration_report", JsonValue::CreateStringValue(L"synthetic-evidence.txt"));
        json.Insert(L"calibration_report_sha256", JsonValue::CreateStringValue(winrt::to_hstring(sha256_file(report))));
        std::string model_hash(64, 'a');
        json.Insert(L"calibration_model_sha256", JsonValue::CreateStringValue(winrt::to_hstring(model_hash)));
        auto config = load();
        verify_deployment_evidence(config, model_hash); // Integrity only, never proof of calibration quality.
        rejects([&] { verify_deployment_evidence(config, std::string(64, 'b')); });
        {
            std::ofstream out(report);
            out << "changed";
        }
        rejects([&] { verify_deployment_evidence(config, model_hash); });
        json.SetNamedValue(L"recognition_threshold", JsonValue::CreateNullValue());
        rejects(load);
        json.SetNamedValue(L"recognition_threshold", JsonValue::CreateNumberValue(.75));
        json.SetNamedValue(L"allow_loopback_http", JsonValue::CreateBooleanValue(true));
        rejects(load);
        json.SetNamedValue(L"mode", JsonValue::CreateStringValue(L"deploymnt"));
        rejects(load);
        std::cout << "Development/deployment separation, incomplete config and report integrity: PASS\n";
        return 0;
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
    catch (winrt::hresult_error const &e)
    {
        std::cerr << "Windows error " << e.code().value << '\n';
        return 1;
    }
}
