// Запуск:  forge_qa            — все тесты, каждый в своём процессе
//          forge_qa <имя>      — один тест в текущем процессе
//          forge_qa --list     — список тестов
#include "qa.h"

#include <forge/console.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <string>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#else
#include <sys/wait.h>
#endif

namespace {

int run_one(const std::string& name) {
    for (const qa::Case& c : qa::cases()) {
        if (name == c.name) {
            try {
                c.fn();
                return 0;
            } catch (const std::exception& e) {
                std::cout << e.what();
                return 1;
            }
        }
    }
    std::cout << "нет такого теста";
    return 2;
}

struct Result {
    std::string status;
    std::string message;
    double seconds = 0;
};

Result run_child(const std::string& self, const std::string& name) {
    std::string cmd;
#ifdef __linux__
    cmd = "timeout 300 ";
#endif
    cmd += "\"" + self + "\" " + name + " 2>&1";
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";  // cmd.exe снимает внешние кавычки — добавляем ещё одни
#endif
    auto start = std::chrono::steady_clock::now();
    FILE* pipe = popen(cmd.c_str(), "r");
    Result r;
    if (!pipe) {
        r.status = "ERROR";
        r.message = "не удалось запустить процесс";
        return r;
    }
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe)) r.message += buffer;
    int status = pclose(pipe);
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
#ifdef _WIN32
    int code = status;
#else
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
    if (code == 0) r.status = "PASS";
    else if (code == 1) r.status = "FAIL";
    else if (code == 124) r.status = "TIMEOUT";
    else r.status = "CRASH (код " + std::to_string(code) + ")";
    while (!r.message.empty() && (r.message.back() == '\n' || r.message.back() == ' '))
        r.message.pop_back();
    return r;
}

} // namespace

int main(int argc, char** argv) {
    forge::enable_utf8_console();
    if (argc > 1 && std::string(argv[1]) == "--list") {
        for (const qa::Case& c : qa::cases()) std::cout << c.group << "\t" << c.name << "\n";
        return 0;
    }
    if (argc > 1) return run_one(argv[1]);

    std::map<std::string, int> counts;
    std::string group;
    for (const qa::Case& c : qa::cases()) {
        if (group != c.group) {
            group = c.group;
            std::cout << "\n=== " << group << " ===\n";
        }
        Result r = run_child(argv[0], c.name);
        counts[r.status.substr(0, r.status.find(' '))]++;
        std::printf("[%-7s] %-38s %6.2fs  %s\n", r.status.c_str(), c.name, r.seconds,
                    c.description);
        if (r.status != "PASS" && !r.message.empty()) {
            std::cout << "          -> " << r.message.substr(0, 400) << "\n";
        }
    }
    std::cout << "\nИтого:";
    for (auto& [status, n] : counts) std::cout << "  " << status << " " << n;
    std::cout << "  (всего " << qa::cases().size() << ")\n";
    return counts["PASS"] == static_cast<int>(qa::cases().size()) ? 0 : 1;
}
