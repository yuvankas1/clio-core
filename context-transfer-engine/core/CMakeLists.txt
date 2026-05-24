cmake_minimum_required(VERSION 3.10)

# Create runtime library using modern ChiMod build functions
add_clio_module_runtime(
  SOURCES
    src/core_runtime.cc
    src/core_config.cc
    src/core_dpe.cc
    src/autogen/core_lib_exec.cc
    # GPU runtime concept removed — kernels submit tasks to the CPU runtime
    # via gpu2cpu_queue and the CPU executes them through the standard
    # chi::Container dispatch path.
)

# Surface the GPU backend flags into the runtime library so the
# GpuMetadataCache helpers (which reach ctp::GpuApi::MallocManaged etc.)
# see the matching #if CTP_ENABLE_{CUDA,ROCM,SYCL} branches. Without
# these the GPU code paths in core_runtime.cc are no-ops.
if(CLIO_CORE_ENABLE_SYCL)
  target_compile_definitions(clio_cte_core_runtime PRIVATE CTP_ENABLE_SYCL=1)
  if(EXISTS "/opt/intel/dpcpp/include/sycl/sycl.hpp")
    target_include_directories(clio_cte_core_runtime PRIVATE
      /opt/intel/dpcpp/include
      /opt/intel/dpcpp/include/sycl)
  elseif(EXISTS "/opt/adaptivecpp/include/AdaptiveCpp/sycl/sycl.hpp")
    target_include_directories(clio_cte_core_runtime PRIVATE
      /opt/adaptivecpp/include/AdaptiveCpp)
  endif()
endif()
if(CLIO_CORE_ENABLE_CUDA)
  target_compile_definitions(clio_cte_core_runtime PRIVATE CTP_ENABLE_CUDA=1)
endif()
if(CLIO_CORE_ENABLE_ROCM)
  target_compile_definitions(clio_cte_core_runtime PRIVATE CTP_ENABLE_ROCM=1)
endif()

# Create client library using modern ChiMod build functions
add_clio_module_client(
  SOURCES
    src/core_client.cc
    src/content_transfer_engine.cc
    src/tag.cc
)

# Tell clio_cte/api.h to emit dllexport (instead of the default dllimport
# that dependent test binaries / adapter DLLs see) for data globals like
# g_cte_client.  Mirrors CLIO_RUN_BUILDING_DLL on the clio_run_cxx target.
if(TARGET clio_cte_core_client)
  target_compile_definitions(clio_cte_core_client PRIVATE CLIO_CTE_CORE_BUILDING_DLL=1)
endif()

# Knowledge graph / Acropolis compile definitions. KG feature work was
# originally on 254-investigate-cuda-uvm under WRP_CTE_* names; the branch-
# merge into semantic-search renamed them to CLIO_CTE_*. No legacy WRP_CTE_KG_*
# alias is provided.
#
# core_runtime.h includes the KG headers (depth_controller, kg_backend, etc.)
# unconditionally, so httplib + nlohmann_json must be on the include path even
# when the CLIO_CTE_ENABLE_KNOWLEDGE_GRAPH compile definition is off — the
# headers must at least *parse*.  cpp-httplib is header-only.
target_include_directories(clio_cte_core_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/httplib>)
target_include_directories(clio_cte_core_client PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/httplib>)
target_link_libraries(clio_cte_core_runtime PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(clio_cte_core_client PUBLIC nlohmann_json::nlohmann_json)

if(CLIO_CTE_ENABLE_KNOWLEDGE_GRAPH)
  message(STATUS "CTE Core: Knowledge graph ENABLED")
  target_compile_definitions(clio_cte_core_runtime PUBLIC CLIO_CTE_ENABLE_KNOWLEDGE_GRAPH)
  target_compile_definitions(clio_cte_core_client PUBLIC CLIO_CTE_ENABLE_KNOWLEDGE_GRAPH)

  # Pluggable KG backend compile definitions
  if(CLIO_CTE_KG_ELASTICSEARCH)
    target_compile_definitions(clio_cte_core_runtime PRIVATE CLIO_CTE_KG_ELASTICSEARCH)
  endif()
  if(CLIO_CTE_KG_NEO4J)
    target_compile_definitions(clio_cte_core_runtime PRIVATE CLIO_CTE_KG_NEO4J)
  endif()
  if(CLIO_CTE_KG_QDRANT)
    target_compile_definitions(clio_cte_core_runtime PRIVATE CLIO_CTE_KG_QDRANT)
  endif()

  # Acropolis L2 HDF5 extractor — optional, only when HDF5 is available.
  if(CLIO_CORE_ENABLE_HDF5)
    find_package(HDF5 QUIET COMPONENTS C)
    if(HDF5_FOUND)
      target_compile_definitions(clio_cte_core_runtime PRIVATE CLIO_CORE_ENABLE_HDF5)
      target_include_directories(clio_cte_core_runtime PRIVATE ${HDF5_INCLUDE_DIRS})
      # Use PUBLIC so downstream test binaries that link the runtime .so also
      # resolve libhdf5's transitive references.
      target_link_libraries(clio_cte_core_runtime PUBLIC ${HDF5_C_LIBRARIES})

      # HDF5's REST VOL needs libcurl with matching symbol versions. The
      # system libhdf5 (apt) was built against the system libcurl
      # (CURL_OPENSSL_4 versioned symbols), while find_package(CURL) may pick
      # the conda curl which uses different versions. Link the system curl
      # directly by path when HDF5 came from /usr/lib.
      set(_system_curl "/usr/lib/x86_64-linux-gnu/libcurl.so.4")
      if(EXISTS "${_system_curl}" AND HDF5_C_LIBRARIES MATCHES "/usr/lib/")
        target_link_libraries(clio_cte_core_runtime PUBLIC "${_system_curl}")
        message(STATUS "CTE Core: HDF5 L2 extractor enabled (system curl: ${_system_curl})")
      else()
        find_package(CURL)
        if(CURL_FOUND)
          target_link_libraries(clio_cte_core_runtime PUBLIC CURL::libcurl)
          message(STATUS "CTE Core: HDF5 L2 extractor enabled (found curl: ${CURL_LIBRARIES})")
        else()
          message(WARNING "CTE Core: HDF5 L2 extractor enabled but curl not found")
        endif()
      endif()
    else()
      message(STATUS "CTE Core: HDF5 L2 extractor requested but HDF5 not found")
    endif()
  endif()
else()
  message(STATUS "CTE Core: Knowledge graph disabled (use -DCLIO_CTE_ENABLE_KNOWLEDGE_GRAPH=ON to enable)")
endif()

# Note: Cross-namespace dependencies (clio_run admin and bdev) are automatically
# linked by add_clio_module_* functions in ClioCoreCommon.cmake for non-clio_run namespaces
