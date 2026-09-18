#include "support/edge_fixtures.hpp"
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
