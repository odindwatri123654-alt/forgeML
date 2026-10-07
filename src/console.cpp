#include <forge/console.h>

#ifdef _WIN32
#define NOMINMAX  // иначе windows.h определит макросы min/max и сломает forge::max
#include <windows.h>
#endif

namespace forge {

void enable_utf8_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
}

} // namespace forge
