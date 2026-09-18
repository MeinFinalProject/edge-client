#pragma once
#include "domain/attendance_record.hpp"
#include "domain/gallery_snapshot.hpp"
namespace edge_app
{
AttendanceRecord envelope(AttendanceObservation const &event, std::string const &device,
                          std::string const &gallery_version);
GallerySnapshot parse_gallery(std::string const &json, std::string const &etag, std::string const &model_hash);
// Offline import uses the same validator, with the ETag carried in the file.
GallerySnapshot parse_gallery_file(std::string const &json, std::string const &model_hash);
std::string batch_payload(std::string const &device, std::vector<AttendanceRecord> const &records);
std::vector<DeliveryReceipt> parse_receipts(std::string const &json, std::vector<AttendanceRecord> const &sent);
} // namespace edge_app
