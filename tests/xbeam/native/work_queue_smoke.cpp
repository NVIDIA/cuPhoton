// SPDX-License-Identifier: Apache-2.0
#include "xbeam/work_queue.hpp"
#include <atomic>
#include <stdexcept>
int main() {
    std::atomic<int> completed{0};
    {
        xbeam::WorkQueue workers(2);
        std::promise<void> released;
        auto gate = released.get_future().share();
        auto first = workers.submit([&] {
            gate.wait();
            ++completed;
        });
        auto second = workers.submit([&] {
            ++completed;
            released.set_value();
        });
        first.get();
        second.get();
        auto failed =
            workers.submit([] { throw std::runtime_error("worker failure"); });
        bool caught = false;
        try {
            failed.get();
        } catch (const std::runtime_error&) {
            caught = true;
        }
        if (!caught) {
            return 1;
        }
        for (int i = 0; i < 100; ++i) {
            workers.submit([&] { ++completed; });
        }
        // Scope exit must drain submitted work before its inputs are released.
    }
    if (completed != 102) {
        return 2;
    }
    bool rejected = false;
    try {
        xbeam::WorkQueue invalid(0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    return rejected ? 0 : 3;
}
