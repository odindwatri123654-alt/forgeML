#pragma once

namespace forge {

// Windows: переключает консоль в UTF-8, чтобы русский текст не превращался в "кракозябры".
// На Linux/macOS ничего не делает (там консоль и так в UTF-8).
void enable_utf8_console();

} // namespace forge
