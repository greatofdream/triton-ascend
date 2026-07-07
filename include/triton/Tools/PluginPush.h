// Push-based plugin registration extension (not in upstream triton 3.7).
//
// Upstream uses a pull model: loadPlugins() dlopens .so files listed in
// TRITON_PLUGIN_PATHS and calls tritonGetPluginInfo(). This header adds a
// push model for extensions loaded via Python `import` (which triggers a
// pybind PYBIND11_MODULE): the extension calls triton_register_plugin() to
// push its PluginInfo into the main module, where per-category hooks
// (installed by init_triton_ir / init_triton_passes / load_dialects) apply
// the ops / passes / dialects to the main module's pybind classes / pass
// manager / dialect registry.
//
// Both models share the same PluginInfo / OpInfo / PassInfo / DialectInfo
// structs (defined in PluginUtils.h) and the same tritonGetPluginInfo()
// entry point, so an extension built for the push model can also be loaded
// by upstream's pull model once TRITON_PLUGIN_PATHS points at it.

#ifndef TRITON_PLUGIN_PUSH_H
#define TRITON_PLUGIN_PUSH_H

#include "triton/Tools/PluginUtils.h"
#include <functional>

namespace mlir::triton::plugin {

/// Push a plugin's PluginInfo into the main module. If the per-category hooks
/// are already installed, ops / passes / dialects are applied immediately;
/// otherwise the info is queued and flushed when the hooks are installed.
__attribute__((visibility("default")))
void triton_register_plugin(PluginInfo *info);

/// Per-category hooks installed by the main module (init_triton_ir /
/// init_triton_passes / load_dialects). Installing a hook also flushes any
/// plugins registered before the hook was set.
__attribute__((visibility("default")))
void set_op_registration_hook(std::function<void(const OpInfo &)> hook);
__attribute__((visibility("default")))
void set_pass_registration_hook(std::function<void(const PassInfo &)> hook);
__attribute__((visibility("default")))
void set_dialect_registration_hook(
    std::function<void(const DialectInfo &)> hook);

/// Returns all plugins registered so far via triton_register_plugin(). Used
/// by load_dialects to iterate dialect registrations at context-creation
/// time.
__attribute__((visibility("default")))
const std::vector<PluginInfo *> &get_registered_plugins();

} // namespace mlir::triton::plugin

#endif // TRITON_PLUGIN_PUSH_H
