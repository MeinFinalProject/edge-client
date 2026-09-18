#pragma once
#include "domain/uuid_v7.hpp"
#include "security/device_credential.hpp"
#include "storage/gallery_repository.hpp"
#include "storage/outbox_repository.hpp"
#include "sync/retry_policy.hpp"
#include "sync/sync_service.hpp"
#include "sync/wire_format.hpp"
#include <Windows.h>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <wincrypt.h>
#include <winrt/Windows.Data.Json.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winsqlite/winsqlite3.h>

using namespace edge_app;
using namespace winrt::Windows::Data::Json;
namespace edge_test
{
struct TestDatabase
{
    Database database;
    OutboxRepository outbox;
    GalleryRepository gallery;
    explicit TestDatabase(std::filesystem::path const &file) : database(file), outbox(database), gallery(database)
    {
    }
};
inline void check(bool condition, char const *why)
{
    if (!condition)
        throw std::runtime_error(why);
}
template <class F> void rejects(F f)
{
    bool caught = false;
    try
    {
        f();
    }
    catch (...)
    {
        caught = true;
    }
    check(caught, "Expected rejection");
}
inline AttendanceRecord record()
{
    AttendanceObservation e;
    e.occurred_at_utc_ms = utc_ms();
    e.identity_id = "S01";
    e.track_id = 9007199254740993ULL;
    return envelope(e, "device-test", "v1");
}
inline GallerySnapshot gallery()
{
    GallerySnapshot g;
    g.version = "v1";
    g.etag = "\"v1\"";
    g.model_id = "insightface/w600k_r50";
    g.model_sha256 = "test-hash";
    g.template_ids = {"t1"};
    GalleryTemplate t;
    t.identity_id = "S01";
    t.embedding[0] = 1;
    g.templates = {t};
    return g;
}
inline std::string empty_gallery(std::string version = "v2", std::string hash = "test-hash")
{
    return "{\"schema_version\":1,\"gallery_version\":\"" + version +
           "\",\"embedding_model\":\"insightface/w600k_r50\",\"model_sha256\":\"" + hash +
           "\",\"embedding_dimension\":512,\"embedding_encoding\":\"f32le-base64\",\"templates\":[]}";
}
struct Mock : Transport
{
    int code = 200, post_calls = 0, get_calls = 0;
    bool lose_ack = false, partial = false, invalid = false;
    std::string gallery_body, token, retry_after;
    std::unordered_set<std::string> accepted;
    void set_token(std::string const &t) override
    {
        token = t;
    }
    HttpResponse request(std::string const &method, std::string const &, std::string const &body,
                         std::string const &) override
    {
        if (method == "GET")
        {
            ++get_calls;
            return gallery_body.empty() ? HttpResponse{304} : HttpResponse{200, gallery_body, "\"v2\""};
        }
        ++post_calls;
        if (code != 200)
            return {code, "", "", retry_after};
        auto events = JsonObject::Parse(winrt::to_hstring(body)).GetNamedArray(L"events");
        JsonArray results;
        for (auto v : events)
        {
            auto id = v.GetObject().GetNamedString(L"event_id");
            auto fresh = accepted.insert(winrt::to_string(id)).second;
            if (partial && results.Size())
                continue;
            JsonObject ack;
            ack.Insert(L"event_id", JsonValue::CreateStringValue(invalid ? L"unknown" : id));
            ack.Insert(L"status", JsonValue::CreateStringValue(fresh ? L"accepted" : L"duplicate"));
            results.Append(ack);
        }
        if (lose_ack)
        {
            lose_ack = false;
            throw std::runtime_error("Simulated server commit then lost ACK");
        }
        JsonObject response;
        response.Insert(L"schema_version", JsonValue::CreateNumberValue(1));
        response.Insert(L"results", results);
        return {200, winrt::to_string(response.Stringify())};
    }
};
} // namespace edge_test
