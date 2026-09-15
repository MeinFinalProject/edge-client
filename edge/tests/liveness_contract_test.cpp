#include "liveness/liveness.hpp"
#include "liveness/vote.hpp"
#include "pipeline/biometric_attendance_runtime.hpp"
#include "../tools/capture_support.hpp"
#include <iostream>
#include <limits>
#include <type_traits>

using namespace vision_runtime::liveness;
void require(bool value, char const* message) { if (!value) throw std::runtime_error(message); }
int main() {
    try {
        static_assert(std::is_same_v<decltype(vision_runtime::pipeline::BiometricAttendanceConfig{}.pad), LivenessConfig>);
        require(LivenessConfig{}.live_threshold == 0.95F, "TA threshold drifted");
        require(vision_runtime::pipeline::BiometricAttendanceConfig{}.pad.live_threshold == kTaLiveThreshold,
                "Attendance and model thresholds disagree");
        require(classify_score(0.95F) == LivenessDecision::Live, "Equality must pass");
        require(classify_score(std::nextafter(0.95F, 0.0F)) == LivenessDecision::Spoof, "Below threshold passed");
        require(classify_score(1) == LivenessDecision::Live && classify_score(0) == LivenessDecision::Spoof,
                "Score boundaries wrong");
        for (float bad : {-1.F, 2.F, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            bool rejected = false;
            try { (void)classify_score(bad); } catch (std::invalid_argument const&) { rejected = true; }
            require(rejected, "Invalid score treated as evidence");
        }
        VoteCounts votes;
        require(!votes.add(LivenessDecision::InputRejected), "Crop rejection counted as a vote");
        require(votes.attempts == 0 && votes.real == 0 && votes.spoof == 0 && votes.input_rejected == 1,
                "Crop rejection contaminated vote counters");
        require(votes.add(LivenessDecision::Live), "Live vote missing");
        require(!votes.add(LivenessDecision::InputRejected), "Interleaved rejection counted");
        require(votes.add(LivenessDecision::Spoof) && votes.add(LivenessDecision::Live), "Votes missing");
        require(votes.attempts == 3 && votes.real == 2 && votes.spoof == 1 && votes.input_rejected == 2,
                "Temporal vote contaminated by rejected inputs");
        capture::validate_label("real", "live");
        for (auto s : {"photo","display","replay"}) capture::validate_label("spoof",s);
        bool bad_label=false;
        try { capture::validate_label("real","display"); } catch(std::invalid_argument const&) { bad_label=true; }
        require(bad_label, "Contradictory capture label accepted");
        require(capture::csv_line({"", "a,b", "say \"real\""}) == "\"\",\"a,b\",\"say \"\"real\"\"\"\r\n", "CSV escaping broken");
        std::cout << "Selected model threshold, voting, rejection, capture metadata and CSV PASS\n";
        return 0;
    } catch(std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
}
