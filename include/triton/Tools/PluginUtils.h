#pragma once

#include <functional>
#include <vector>

namespace mlir {
class PassManager;
class DialectRegistry;
class Value;
} // namespace mlir

class TritonOpBuilder;

namespace triton::plugin {

// API version; bumped on ABI-breaking changes. Mirrors triton 3.7.
#define TRITON_PLUGIN_API_VERSION 2

// Force-export symbols so extensions linked against libtriton.so can resolve
// them even when libtriton is built with -fvisibility=hidden /
// --exclude-libs,ALL.
#define TRITON_PLUGIN_API __attribute__((visibility("default")))

// Callbacks invoked by the main module to apply plugin-provided ops / passes
// / dialects. Mirrors triton 3.7's PluginUtils.h callback types.
using AddOpCallback = void (*)(TritonOpBuilder &,
                               std::vector<mlir::Value> &);
using AddPassCallback = void (*)(mlir::PassManager *,
                                 const std::vector<std::string> &);
using RegisterDialectCallback = void (*)(mlir::DialectRegistry *);

// Metadata for a single op / pass / dialect provided by a plugin.
struct OpInfo {
  const char *name;
  AddOpCallback addOp;
};

struct PassInfo {
  const char *name;
  AddPassCallback addPass;
};

struct DialectInfo {
  RegisterDialectCallback registerDialect;
};

// Container returned by a plugin's tritonGetPluginInfo() entry point.
// Layout follows triton 3.7 so a plugin built against this header can be
// loaded by triton 3.7's loadPlugins() unchanged (given TRITON_PLUGIN_PATHS).
struct PluginInfo {
  int apiVersion;
  const char *name;
  const char *version;
  const OpInfo *ops;
  int numOps;
  const PassInfo *passes;
  int numPasses;
  const DialectInfo *dialects;
  int numDialects;
  const char *tritonVersion;
};

// Standard entry point a plugin exports. libtriton provides a weak default
// (returns nullptr) so extensions can be absent; a real plugin overrides it
// with a strong symbol.
extern "C" TRITON_PLUGIN_API PluginInfo *tritonGetPluginInfo();

// --- Extension-facing API (called by libtriton_dist etc.) ---
//
// Push a plugin's PluginInfo into the main module. If the main module's
// registration hook is already set (init_triton_ir has run), the ops / passes
// / dialects are applied immediately. Otherwise the request is queued and
// flushed when the hook is installed.
//
// This lets an extension be loaded via Python `import` (which triggers its
// PYBIND11_MODULE) without requiring TRITON_PLUGIN_PATHS / loadPlugins()
// dlopen path — while keeping the tritonGetPluginInfo() entry point
// compatible with triton 3.7's standard plugin mechanism for future
// migration.
TRITON_PLUGIN_API void triton_register_plugin(PluginInfo *info);

// --- Main-module-facing API (called by python/src/ir.cc, passes.cc) ---
//
// Install per-category hooks. Each hook receives the corresponding entries
// from a PluginInfo when triton_register_plugin is called (or, if the plugin
// was registered before the hook was set, the hook is invoked immediately
// upon installation). This separation is needed because ops / passes /
// dialects are applied at different points in the main module:
//   - ops   → init_triton_ir   (ir.cc, on the TritonOpBuilder pybind class)
//   - passes→ init_triton_passes (passes.cc, on the passes submodule)
//   - dialects → load_dialects  (ir.cc, on the DialectRegistry)
TRITON_PLUGIN_API void
set_op_registration_hook(std::function<void(const OpInfo &)> hook);

TRITON_PLUGIN_API void
set_pass_registration_hook(std::function<void(const PassInfo &)> hook);

TRITON_PLUGIN_API void
set_dialect_registration_hook(std::function<void(const DialectInfo &)> hook);

// Returns all plugins registered so far. Used by load_dialects to iterate
// dialect registrations at context-creation time (which happens after all
// plugins have been imported).
TRITON_PLUGIN_API const std::vector<PluginInfo *> &get_registered_plugins();

} // namespace triton::plugin
