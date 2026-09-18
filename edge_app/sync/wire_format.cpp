#include "sync/wire_format.hpp"
#include "domain/uuid_v7.hpp"
#include <Windows.h>
#include <bit>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <wincrypt.h>
#include <winrt/Windows.Data.Json.h>
#include <winrt/Windows.Foundation.Collections.h>
namespace edge_app
{
using namespace winrt::Windows::Data::Json;
namespace
{
std::string str(JsonObject const &o, wchar_t const *key)
{
    return winrt::to_string(o.GetNamedString(key));
}
void text(JsonObject const &o, wchar_t const *key, std::string const &v)
{
    o.Insert(key, JsonValue::CreateStringValue(winrt::to_hstring(v)));
}
void number(JsonObject const &o, wchar_t const *key, double v)
{
    if (!std::isfinite(v))
        throw std::invalid_argument("Non-finite record field");
    o.Insert(key, JsonValue::CreateNumberValue(v));
}
void id(std::string const &s)
{
    if (s.empty() || s.size() > 256 || s.find_first_of("\r\n\0", 0, 3) != std::string::npos)
        throw std::invalid_argument("Invalid identifier");
}
std::string timestamp(std::int64_t ms)
{
    if (ms <= 0)
        throw std::invalid_argument("Occurrence time required");
    __time64_t sec = ms / 1000;
    std::tm t{};
    if (_gmtime64_s(&t, &sec))
        throw std::invalid_argument("UTC time out of range");
    std::ostringstream o;
    o << std::put_time(&t, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3) << std::setfill('0') << ms % 1000 << 'Z';
    return o.str();
}
} // namespace
AttendanceRecord envelope(AttendanceObservation const &e, std::string const &device, std::string const &version)
{
    id(device);
    id(version);
    id(e.identity_id);
    AttendanceRecord r;
    r.occurred_ms = e.occurred_at_utc_ms;
    r.id = uuid_v7(r.occurred_ms);
    JsonObject o;
    number(o, L"schema_version", 1);
    text(o, L"event_id", r.id);
    text(o, L"occurred_at", timestamp(r.occurred_ms));
    text(o, L"device_id", device);
    text(o, L"identity_id", e.identity_id);
    text(o, L"gallery_version", version);
    text(o, L"runtime_version", "edge-v1");
    // Decimal strings retain uint64 precision across JavaScript/JSON consumers.
    text(o, L"track_id", std::to_string(e.track_id));
    text(o, L"camera_frame_id", std::to_string(e.camera_frame_id));
    number(o, L"similarity", e.similarity);
    number(o, L"similarity_margin", e.similarity_margin);
    number(o, L"pad_median_p_real", e.pad_median_p_real);
    r.payload = winrt::to_string(o.Stringify());
    return r;
}
GallerySnapshot parse_gallery(std::string const &document, std::string const &etag, std::string const &expected_hash)
{
    if (document.size() > 32 * 1024 * 1024)
        throw std::invalid_argument("GallerySnapshot exceeds size limit");
    auto o = JsonObject::Parse(winrt::to_hstring(document));
    GallerySnapshot g;
    if (o.GetNamedNumber(L"schema_version") != 1 || o.GetNamedNumber(L"embedding_dimension") != 512 ||
        str(o, L"embedding_encoding") != "f32le-base64")
        throw std::invalid_argument("Unsupported gallery schema/encoding");
    g.model_id = str(o, L"embedding_model");
    g.model_sha256 = str(o, L"model_sha256");
    g.version = str(o, L"gallery_version");
    g.etag = etag;
    id(g.version);
    if (g.etag.empty() || g.etag.size() > 256 || g.etag.find_first_of("\r\n") != std::string::npos)
        throw std::invalid_argument("Invalid gallery ETag");
    if (g.model_id != "insightface/w600k_r50" || g.model_sha256 != expected_hash)
        throw std::invalid_argument("Incompatible gallery model");
    auto list = o.GetNamedArray(L"templates");
    if (list.Size() > 10000)
        throw std::invalid_argument("Too many templates");
    std::unordered_set<std::string> unique;
    for (auto const &v : list)
    {
        auto t = v.GetObject();
        auto tid = str(t, L"template_id");
        id(tid);
        if (!unique.insert(tid).second)
            throw std::invalid_argument("Duplicate template ID");
        GalleryTemplate item;
        item.identity_id = str(t, L"identity_id");
        id(item.identity_id);
        if (t.GetNamedNumber(L"dimension") != 512 || str(t, L"encoding") != "f32le-base64")
            throw std::invalid_argument("Invalid template encoding");
        auto data = str(t, L"data");
        if (data.size() != 2732)
            throw std::invalid_argument("Invalid embedding length");
        std::array<BYTE, 2048> bytes{};
        DWORD size = 2048;
        if (!CryptStringToBinaryA(data.data(), static_cast<DWORD>(data.size()),
                                  CRYPT_STRING_BASE64 | CRYPT_STRING_STRICT, bytes.data(), &size, nullptr, nullptr) ||
            size != 2048)
            throw std::invalid_argument("Invalid embedding base64");
        static_assert(std::endian::native == std::endian::little);
        std::memcpy(item.embedding.data(), bytes.data(), 2048);
        for (float x : item.embedding)
            if (!std::isfinite(x))
                throw std::invalid_argument("Invalid embedding number");
        item.embedding = normalize_embedding(item.embedding);
        g.templates.push_back(std::move(item));
        g.template_ids.push_back(std::move(tid));
    }
    return g;
}
std::string batch_payload(std::string const &device, std::vector<AttendanceRecord> const &rows)
{
    JsonObject body;
    text(body, L"device_id", device);
    number(body, L"schema_version", 1);
    JsonArray events;
    for (auto const &r : rows)
        events.Append(JsonObject::Parse(winrt::to_hstring(r.payload)));
    body.Insert(L"events", events);
    return winrt::to_string(body.Stringify());
}
std::vector<DeliveryReceipt> parse_receipts(std::string const &json, std::vector<AttendanceRecord> const &sent)
{
    auto o = JsonObject::Parse(winrt::to_hstring(json));
    if (o.GetNamedNumber(L"schema_version") != 1)
        throw std::invalid_argument("Invalid ACK schema");
    std::unordered_set<std::string> allowed, seen;
    for (auto const &r : sent)
        allowed.insert(r.id);
    std::vector<DeliveryReceipt> result;
    auto list = o.GetNamedArray(L"results");
    if (list.Size() > sent.size())
        throw std::invalid_argument("Too many ACKs");
    for (auto v : list)
    {
        auto a = v.GetObject();
        DeliveryReceipt r{str(a, L"event_id"), str(a, L"status")};
        if (!allowed.contains(r.event_id) || !seen.insert(r.event_id).second ||
            (r.disposition != "accepted" && r.disposition != "duplicate" && r.disposition != "rejected"))
            throw std::invalid_argument("Invalid ACK member");
        result.push_back(std::move(r));
    }
    return result;
}

GallerySnapshot parse_gallery_file(std::string const &json, std::string const &hash)
{
    auto document = JsonObject::Parse(winrt::to_hstring(json));
    return parse_gallery(json, winrt::to_string(document.GetNamedString(L"etag")), hash);
}

} // namespace edge_app
