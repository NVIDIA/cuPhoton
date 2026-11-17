// SPDX-License-Identifier: Apache-2.0
#include "xbeam/radio.hpp"
#include "xbeam/recording.hpp"
#include <iomanip>
#include <iostream>
int main(int argc, char** argv) {
    if (argc == 10 && std::string(argv[1]) == "inspect-replay") {
        try {
            xbeam::RadioJob job(argv[2]);
            xbeam::RadioRecording recording(job, std::stoi(argv[3]), argv[5]);
            auto plan = recording.plan(
                argv[6],
                std::stoi(argv[4]),
                std::stoll(argv[7]),
                std::stoll(argv[8]),
                std::stoll(argv[9]));
            const auto& g = recording.geometry();
            std::cout << "{\"job_slot\":" << g.job_slot
                      << ",\"allocation_frames\":"
                      << g.windows.allocation_frames << ",\"rows\":[";
            bool comma = false;
            for (const auto& r : plan.replay) {
                if (comma) {
                    std::cout << ',';
                }
                comma = true;
                std::cout << '[' << r.fft << ',' << r.physical_start << ','
                          << r.timing_displacement << ',' << r.valid_first
                          << ',' << r.valid_stop << ',' << int(r.accepted)
                          << ']';
            }
            std::cout << "]}\n";
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << '\n';
            return 1;
        }
    }

    if (argc == 4 && std::string(argv[1]) == "inspect-index") {
        try {
            xbeam::IndexReader reader(argv[2], argv[3]);
            std::cout << reader.metadata_json() << '\n';
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << '\n';
            return 1;
        }
    }

    if (argc != 3
        || (std::string(argv[1]) != "inspect-vdif"
            && std::string(argv[1]) != "inspect-job")) {
        std::cerr
            << "usage: xbeam-native inspect-vdif RECORDING.vdif | inspect-job "
               "JOB.input\n";
        return 2;
    }
    try {
        std::cout
            << (std::string(argv[1]) == "inspect-job"
                       ? xbeam::load_interop_job_json(argv[2])
                       : xbeam::inspect_vdif_json(argv[2]))
            << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
