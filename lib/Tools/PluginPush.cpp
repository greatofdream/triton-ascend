#include "triton/Tools/PluginPush.h"

#include <cstdio>
#include <functional>
#include <mutex>

// ---------------------------------------------------------------------------
// Push-based registration extension (not in upstream triton 3.7).
// ---------------------------------------------------------------------------

namespace {

struct PushState {
  std::mutex mutex;
  std::vector<mlir::triton::plugin::PluginInfo *> registered;
  std::function<void(const mlir::triton::plugin::OpInfo &)> op_hook;
  std::function<void(const mlir::triton::plugin::PassInfo &)> pass_hook;
  std::function<void(const mlir::triton::plugin::DialectInfo &)>
      dialect_hook;
};

PushState &pushState() {
  // Leak intentionally: PushState holds std::function hooks that may capture
  // pybind objects (py::module_). Destroying them at exit crashes because
  // the Python interpreter is already gone (module_dealloc → _Py_GetConfig
  // → SIGSEGV). Using new avoids the exit-time destructor.
  static PushState *s = new PushState();
  return *s;
}

void applyOps(mlir::triton::plugin::PluginInfo *info) {
  auto &s = pushState();
  if (!s.op_hook) {
    fprintf(stderr, "[PLUGIN] applyOps: SKIP (op_hook null) plugin=%s numOps=%zu\n",
            info->pluginName, info->numOps);
    return;
  }
  for (size_t i = 0; i < info->numOps; ++i) {
    fprintf(stderr, "[PLUGIN] applyOps: calling op_hook for op=%s\n",
            info->ops[i].name);
    s.op_hook(info->ops[i]);
  }
}

void applyPasses(mlir::triton::plugin::PluginInfo *info) {
  auto &s = pushState();
  if (!s.pass_hook)
    return;
  for (size_t i = 0; i < info->numPasses; ++i)
    s.pass_hook(info->passes[i]);
}

void applyDialects(mlir::triton::plugin::PluginInfo *info) {
  auto &s = pushState();
  if (!s.dialect_hook)
    return;
  for (size_t i = 0; i < info->numDialects; ++i)
    s.dialect_hook(info->dialects[i]);
}

} // namespace

namespace mlir::triton::plugin {

void triton_register_plugin(PluginInfo *info) {
  if (!info)
    return;
  auto &s = pushState();
  std::lock_guard<std::mutex> lock(s.mutex);
  fprintf(stderr, "[PLUGIN] triton_register_plugin: plugin=%s numOps=%zu numPasses=%zu numDialects=%zu op_hook=%d\n",
          info->pluginName, info->numOps, info->numPasses, info->numDialects,
          (bool)s.op_hook);
  s.registered.push_back(info);
  applyOps(info);
  applyPasses(info);
  applyDialects(info);
}

void set_op_registration_hook(
    std::function<void(const OpInfo &)> hook) {
  auto &s = pushState();
  std::lock_guard<std::mutex> lock(s.mutex);
  fprintf(stderr, "[PLUGIN] set_op_registration_hook: registered=%zu\n",
          s.registered.size());
  s.op_hook = std::move(hook);
  for (auto *info : s.registered)
    applyOps(info);
}

void set_pass_registration_hook(
    std::function<void(const PassInfo &)> hook) {
  auto &s = pushState();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.pass_hook = std::move(hook);
  for (auto *info : s.registered)
    applyPasses(info);
}

void set_dialect_registration_hook(
    std::function<void(const DialectInfo &)> hook) {
  auto &s = pushState();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.dialect_hook = std::move(hook);
  for (auto *info : s.registered)
    applyDialects(info);
}

const std::vector<PluginInfo *> &get_registered_plugins() {
  auto &s = pushState();
  std::lock_guard<std::mutex> lock(s.mutex);
  return s.registered;
}

} // namespace mlir::triton::plugin
