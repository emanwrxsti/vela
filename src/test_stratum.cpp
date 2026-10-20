#include "stratum.h"
#include <chrono>
#include <csignal>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    #if !defined(_WIN32)
    std::signal(SIGPIPE, SIG_IGN);
#endif
    meow::StratumClient c("127.0.0.1", std::stoi(argv[1]), std::stoi(argv[2]),
                         "transport-test", "x", "transport-review/0.1.4");
    std::mutex logs;
    c.log = [&](const std::string& s) {
        if (s.find("share accepted") == std::string::npos) {
            std::lock_guard<std::mutex> l(logs);
            std::cout << s << std::endl;
        }
    };
    c.start();
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(std::stoi(argv[3]));
    std::vector<std::thread> writers;
    for (int i = 0; i < 8; ++i) writers.emplace_back([&, i] {
        unsigned n = 0;
        while (std::chrono::steady_clock::now() < end) {
            meow::Job j;
            if (c.currentJob(j)) {
                // Occasionally finish work after its original session has closed.
                if (++n % 31 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(80));
                c.submit(j, j.extranonce1Hex, "00000001", "00000001",
                         std::string(1024, 'a'), std::string(64, 'b'));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2 + i % 3));
        }
    });
    for (auto& t : writers) t.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    c.stop();
    std::cout << "RESULT {\"connections\":" << c.stats.reconnects.load()
              << ",\"accepted\":" << c.stats.accepted.load()
              << ",\"rejected\":" << c.stats.rejected.load()
              << ",\"submitted\":" << c.stats.submitted.load()
              << ",\"discarded\":" << c.stats.discarded.load() << "}\n";
}
