#include "triton/Tools/PluginUtils.h"

#include <mutex>
#include <vector>

namespace triton::plugin {

namespace {

// Weak default so libtriton links cleanly when no plugin is present.
extern "C" TRITON_PLUGIN_API PluginInfo *tritonGetPluginInfo() {
  return nullptr;
}

struct State {
  std::mutex mutex;
  std::vector<PluginInfo *> registered;
  std::function<void(const OpInfo &)> op_hook;
  std::function<void(const PassInfo &)> pass_hook;
  std::function<void(const DialectInfo &)> dialect_hook;
};

State &state() {
  static State s;
  return s;
}

void apply_ops(PluginInfo *info) {
  auto &s = state();
  if (!s.op_hook)
    return;
  for (int i = 0; i < info->numOps; ++i)
    s.op_hook(info->ops[i]);
}

void apply_passes(PluginInfo *info) {
  auto &s = state();
  if (!s.pass_hook)
    return;
  for (int i = 0; i < info->numPasses; ++i)
    s.pass_hook(info->passes[i]);
}

void apply_dialects(PluginInfo *info) {
  auto &s = state();
  if (!s.dialect_hook)
    return;
  for (int i = 0; i < info->numDialects; ++i)
    s.dialect_hook(info->dialects[i]);
}

} // namespace

void triton_register_plugin(PluginInfo *info) {
  if (!info)
    return;
  auto &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.registered.push_back(info);
  apply_ops(info);
  apply_passes(info);
  apply_dialects(info);
}

void set_op_registration_hook(std::function<void(const OpInfo &)> hook) {
  auto &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.op_hook = std::move(hook);
  for (auto *info : s.registered)
    apply_ops(info);
}

void set_pass_registration_hook(std::function<void(const PassInfo &)> hook) {
  auto &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.pass_hook = std::move(hook);
  for (auto *info : s.registered)
    apply_passes(info);
}

void set_dialect_registration_hook(
    std::function<void(const DialectInfo &)> hook) {
  auto &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.dialect_hook = std::move(hook);
  for (auto *info : s.registered)
    apply_dialects(info);
}

const std::vector<PluginInfo *> &get_registered_plugins() {
  auto &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return s.registered;
}

} // namespace triton::plugin
