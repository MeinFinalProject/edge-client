#include "support/edge_fixtures.hpp"
#include <limits>
using namespace edge_test;
int main()
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        auto root = std::filesystem::current_path() / "edge-test-data" / uuid_v7(utc_ms());
        std::filesystem::create_directories(root);
        rejects([&] { (void)parse_gallery(empty_gallery("v2", "wrong"), "\"v2\"", "test-hash"); });
        rejects([&] { (void)parse_gallery(empty_gallery(), "", "test-hash"); });
        rejects([&] { (void)parse_gallery(empty_gallery(), std::string("tag\0tail", 8), "test-hash"); });
        for (auto bad : {"", "{", "[]", "{}", "null"})
            rejects([&] { (void)parse_gallery(bad, "\"v2\"", "test-hash"); });
        for (auto key : {L"schema_version", L"gallery_version", L"embedding_model", L"model_sha256",
                         L"embedding_dimension", L"embedding_encoding", L"templates"})
        {
            auto doc = JsonObject::Parse(winrt::to_hstring(empty_gallery()));
            doc.Remove(key);
            rejects([&] { (void)parse_gallery(winrt::to_string(doc.Stringify()), "\"v2\"", "test-hash"); });
        }
        auto row = record();
        auto batch = JsonObject::Parse(winrt::to_hstring(batch_payload("device-test", {row})));
        auto serialized = batch.GetNamedArray(L"events").GetObjectAt(0);
        check(serialized.GetNamedString(L"event_id") == winrt::to_hstring(row.id) &&
                  serialized.GetNamedString(L"track_id") == L"9007199254740993" &&
                  serialized.GetNamedString(L"device_id") == batch.GetNamedString(L"device_id"),
              "Batch serialization changed metadata/uint64 precision");
        rejects([&] { (void)batch_payload("another-device", {row}); });
        auto changed = row;
        changed.id = uuid_v7(utc_ms());
        rejects([&] { (void)batch_payload("device-test", {changed}); });
        AttendanceObservation invalid;
        invalid.identity_id = "S01";
        invalid.occurred_at_utc_ms = utc_ms();
        invalid.similarity = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { (void)envelope(invalid, "device-test", "v1"); });
        auto receipt = [&](std::string const &status) {
            return "{\"event_id\":\"" + row.id + "\",\"status\":\"" + status + "\"}";
        };
        for (auto status : {"accepted", "duplicate", "rejected"})
            check(parse_receipts("{\"schema_version\":1,\"results\":[" + receipt(status) + "]}", {row}).size() == 1,
                  "Valid receipt rejected");
        check(parse_receipts("{\"schema_version\":1,\"results\":[]}", {row}).empty(),
              "Missing receipt must remain pending");
        for (auto bad : {"", "{", "{}", "{\"schema_version\":2,\"results\":[]}"})
            rejects([&] { (void)parse_receipts(bad, {row}); });
        rejects([&] {
            (void)parse_receipts("{\"schema_version\":1,\"results\":[" + receipt("accepted") + "," +
                                     receipt("accepted") + "]}",
                                 {row, record()});
        });
        rejects(
            [&] { (void)parse_receipts("{\"schema_version\":1,\"results\":[" + receipt("unexpected") + "]}", {row}); });
        {
            auto source = gallery().templates[0].embedding;
            DWORD size = 0;
            check(CryptBinaryToStringA(reinterpret_cast<BYTE *>(source.data()), 2048,
                                       CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF, nullptr, &size),
                  "Base64 length failed");
            std::string encoded(size, '\0');
            check(CryptBinaryToStringA(reinterpret_cast<BYTE *>(source.data()), 2048,
                                       CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF, encoded.data(), &size),
                  "Base64 encoding failed");
            encoded.resize(size);
            if (!encoded.empty() && encoded.back() == '\0')
                encoded.pop_back();
            auto doc = JsonObject::Parse(winrt::to_hstring(empty_gallery()));
            JsonObject t;
            t.Insert(L"template_id", JsonValue::CreateStringValue(L"t1"));
            t.Insert(L"identity_id", JsonValue::CreateStringValue(L"S01"));
            t.Insert(L"dimension", JsonValue::CreateNumberValue(512));
            t.Insert(L"encoding", JsonValue::CreateStringValue(L"f32le-base64"));
            t.Insert(L"data", JsonValue::CreateStringValue(winrt::to_hstring(encoded)));
            JsonArray list;
            list.Append(t);
            doc.Insert(L"templates", list);
            auto parsed = parse_gallery(winrt::to_string(doc.Stringify()), "\"v2\"", "test-hash");
            check(parsed.templates.size() == 1 && parsed.templates[0].embedding[0] == 1,
                  "Embedding byte contract failed");
            for (auto key : {L"template_id", L"identity_id", L"dimension", L"encoding", L"data"})
            {
                auto incomplete = JsonObject::Parse(doc.Stringify());
                incomplete.GetNamedArray(L"templates").GetObjectAt(0).Remove(key);
                rejects([&] { (void)parse_gallery(winrt::to_string(incomplete.Stringify()), "\"v2\"", "test-hash"); });
            }
            for (float value : {0.F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
            {
                auto bad_source = source;
                bad_source[0] = value;
                DWORD count = 0;
                CryptBinaryToStringA(reinterpret_cast<BYTE *>(bad_source.data()), 2048,
                                     CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF, nullptr, &count);
                std::string bad_encoded(count, '\0');
                check(CryptBinaryToStringA(reinterpret_cast<BYTE *>(bad_source.data()), 2048,
                                           CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF, bad_encoded.data(), &count),
                      "Encode bad embedding failed");
                if (!bad_encoded.empty() && bad_encoded.back() == '\0')
                    bad_encoded.pop_back();
                auto corrupt = JsonObject::Parse(doc.Stringify());
                corrupt.GetNamedArray(L"templates")
                    .GetObjectAt(0)
                    .SetNamedValue(L"data", JsonValue::CreateStringValue(winrt::to_hstring(bad_encoded)));
                rejects([&] { (void)parse_gallery(winrt::to_string(corrupt.Stringify()), "\"v2\"", "test-hash"); });
            }
            list.Append(t);
            rejects([&] { (void)parse_gallery(winrt::to_string(doc.Stringify()), "\"v2\"", "test-hash"); });
            list.RemoveAtEnd();
            t.SetNamedValue(L"data", JsonValue::CreateStringValue(winrt::to_hstring(std::string(2732, '!'))));
            rejects([&] { (void)parse_gallery(winrt::to_string(doc.Stringify()), "\"v2\"", "test-hash"); });
        }
        std::cout << "Gallery wire format PASS\n";
        return 0;
    }
    catch (winrt::hresult_error const &e)
    {
        std::cerr << "WinRT error " << std::hex << e.code().value << '\n';
        return 1;
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
