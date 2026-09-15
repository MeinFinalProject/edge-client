#include "camera/directshow_low_light_control.hpp"

#include <cwchar>
#include <sstream>

#include <dshow.h>
#include <ks.h>
#include <ksmedia.h>
#include <ocidl.h>
#include <oleauto.h>
#include <winrt/base.h>

namespace vision_runtime::detail {
namespace {

constexpr long kLowLightProperty =
    static_cast<long>(KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY);

std::string hr_text(HRESULT hr) {
    std::ostringstream out;
    out << "HRESULT=0x" << std::hex << std::uppercase
        << static_cast<unsigned long>(hr);
    return out.str();
}

winrt::com_ptr<IBaseFilter> find_camera_filter(
    std::wstring const& friendly_name
) {
    winrt::com_ptr<ICreateDevEnum> device_enum;
    winrt::check_hresult(
        CoCreateInstance(
            CLSID_SystemDeviceEnum,
            nullptr,
            CLSCTX_INPROC_SERVER,
            __uuidof(ICreateDevEnum),
            device_enum.put_void()
        )
    );

    winrt::com_ptr<IEnumMoniker> enum_moniker;
    HRESULT hr = device_enum->CreateClassEnumerator(
        CLSID_VideoInputDeviceCategory,
        enum_moniker.put(),
        0
    );

    if (hr == S_FALSE || !enum_moniker) {
        return nullptr;
    }
    winrt::check_hresult(hr);

    winrt::com_ptr<IMoniker> moniker;
    ULONG fetched = 0;

    while (enum_moniker->Next(1, moniker.put(), &fetched) == S_OK) {
        winrt::com_ptr<IPropertyBag> property_bag;
        HRESULT bag_hr = moniker->BindToStorage(
            nullptr,
            nullptr,
            __uuidof(IPropertyBag),
            property_bag.put_void()
        );

        if (SUCCEEDED(bag_hr) && property_bag) {
            VARIANT value;
            VariantInit(&value);

            HRESULT read_hr = property_bag->Read(
                L"FriendlyName",
                &value,
                nullptr
            );

            if (
                SUCCEEDED(read_hr) &&
                value.vt == VT_BSTR &&
                value.bstrVal != nullptr
            ) {
                std::wstring current_name(value.bstrVal);

                if (_wcsicmp(current_name.c_str(), friendly_name.c_str()) == 0) {
                    winrt::com_ptr<IBaseFilter> filter;
                    HRESULT bind_hr = moniker->BindToObject(
                        nullptr,
                        nullptr,
                        __uuidof(IBaseFilter),
                        filter.put_void()
                    );

                    VariantClear(&value);

                    if (SUCCEEDED(bind_hr)) {
                        return filter;
                    }

                    winrt::check_hresult(bind_hr);
                }
            }

            VariantClear(&value);
        }

        moniker = nullptr;
    }

    return nullptr;
}

winrt::com_ptr<IAMCameraControl> get_camera_control(
    std::wstring const& friendly_name,
    bool& device_found
) {
    auto filter = find_camera_filter(friendly_name);
    if (!filter) {
        device_found = false;
        return nullptr;
    }

    device_found = true;

    winrt::com_ptr<IAMCameraControl> control;
    HRESULT hr = filter->QueryInterface(
        __uuidof(IAMCameraControl),
        control.put_void()
    );

    if (FAILED(hr)) {
        return nullptr;
    }

    return control;
}

} // namespace

LegacyLowLightState query_legacy_low_light_priority(
    std::wstring const& friendly_name
) {
    LegacyLowLightState result;

    try {
        bool device_found = false;
        auto control = get_camera_control(friendly_name, device_found);
        result.device_found = device_found;

        if (!control) {
            result.error = device_found
                ? "Device ditemukan, tetapi IAMCameraControl tidak tersedia"
                : "Device DirectShow dengan FriendlyName yang sama tidak ditemukan";
            return result;
        }

        result.camera_control_available = true;

        long value = 0;
        long flags = 0;
        HRESULT get_hr = control->Get(
            kLowLightProperty,
            &value,
            &flags
        );

        if (FAILED(get_hr)) {
            result.error = "Property 19 tidak didukung: " + hr_text(get_hr);
            return result;
        }

        result.supported = true;
        result.value = value;
        result.flags = flags;

        long minimum = 0;
        long maximum = 0;
        long step = 0;
        long default_value = 0;
        long caps = 0;

        HRESULT range_hr = control->GetRange(
            kLowLightProperty,
            &minimum,
            &maximum,
            &step,
            &default_value,
            &caps
        );

        if (SUCCEEDED(range_hr)) {
            result.minimum = minimum;
            result.maximum = maximum;
            result.step = step;
            result.default_value = default_value;
            result.caps = caps;
        }

    } catch (winrt::hresult_error const& e) {
        result.error = "WinRT/COM error: " + winrt::to_string(e.message());
    } catch (std::exception const& e) {
        result.error = e.what();
    }

    return result;
}

bool set_legacy_low_light_priority(
    std::wstring const& friendly_name,
    long value,
    std::string& error
) {
    try {
        bool device_found = false;
        auto control = get_camera_control(friendly_name, device_found);

        if (!control) {
            error = device_found
                ? "IAMCameraControl tidak tersedia"
                : "Device DirectShow tidak ditemukan";
            return false;
        }

        HRESULT hr = control->Set(
            kLowLightProperty,
            value,
            CameraControl_Flags_Manual
        );

        if (FAILED(hr)) {
            error = "Set property 19 gagal: " + hr_text(hr);
            return false;
        }

        long verify_value = -1;
        long verify_flags = 0;
        hr = control->Get(
            kLowLightProperty,
            &verify_value,
            &verify_flags
        );

        if (FAILED(hr)) {
            error = "SET berhasil tetapi GET verifikasi gagal: " + hr_text(hr);
            return false;
        }

        if (verify_value != value) {
            std::ostringstream message;
            message << "Driver tidak mempertahankan property 19. requested="
                    << value << " observed=" << verify_value;
            error = message.str();
            return false;
        }

        error.clear();
        return true;

    } catch (winrt::hresult_error const& e) {
        error = "WinRT/COM error: " + winrt::to_string(e.message());
        return false;
    } catch (std::exception const& e) {
        error = e.what();
        return false;
    }
}

} // namespace vision_runtime::detail
