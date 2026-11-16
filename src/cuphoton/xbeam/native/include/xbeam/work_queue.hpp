// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>
namespace xbeam {
class WorkQueue {
    std::mutex mutex;
    std::condition_variable ready;
    std::queue<std::function<void()>> pending;
    std::vector<std::thread> workers;
    bool stopping = false;

public:
    explicit WorkQueue(unsigned count) {
        if (count == 0) {
            throw std::invalid_argument("work queue needs at least one worker");
        }
        try {
            for (unsigned k = 0; k < count; ++k) {
                workers.emplace_back([this] {
                    for (;;) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(mutex);
                            ready.wait(lock, [this] {
                                return stopping || !pending.empty();
                            });
                            if (stopping && pending.empty()) {
                                return;
                            }
                            task = std::move(pending.front());
                            pending.pop();
                        }
                        task();
                    }
                });
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
            }
            ready.notify_all();
            for (auto& worker : workers) {
                worker.join();
            }
            throw;
        }
    }
    ~WorkQueue() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto& worker : workers) {
            worker.join();
        }
    }
    std::future<void> submit(std::function<void()> function) {
        auto task =
            std::make_shared<std::packaged_task<void()>>(std::move(function));
        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex);
            pending.emplace([task] { (*task)(); });
        }
        ready.notify_one();
        return future;
    }
};
}  // namespace xbeam
