add_triton_library(TritonTools
  GenericSwizzling.cpp
  LayoutUtils.cpp
  LinearLayout.cpp
  # PluginUtils.cpp is built as a separate STATIC library (see below) so that
  # plugin extensions (e.g. libtriton_dist.so) that add_subdirectory(this) do
  # not accidentally link their own copy of the push-based PushState, which
  # would break the single-shared-state invariant with libtriton.so.

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  f2reduce
)

# STATIC (not OBJECT) so only libtriton.so links it via TRITON_LIBRARIES;
# plugin .so files include only the header (declarations) and resolve
# symbols against libtriton.so at runtime -> single shared PushState.
add_library(TritonPluginUtils STATIC PluginUtils.cpp)
target_link_libraries(TritonPluginUtils PUBLIC MLIRIR MLIRLLVMDialect)
