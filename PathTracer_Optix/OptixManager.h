#ifndef OPTIX_MANAGER_H
#define OPTIX_MANAGER_H

#include "common.h"




#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Trackball.h>
#include <sutil/Camera.h>
#include <sutil/GLDisplay.h>


#include "tinylogger.h"
#include "GLFCallbacks.h"
#include "pathTracer.h"



#include "TinyObjWrapper.h"



template <typename T>
struct SBTRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

using RayGenerationRecord = SBTRecord<RayGenerationData>;
using MissRecord = SBTRecord<MissData>;
using HitGroupRecord = SBTRecord<HitGroupData>;

struct PathTracerState {
  OptixDeviceContext context = nullptr;

  OptixTraversableHandle gas_handle = 0;
  CUdeviceptr d_gas_output_buffer = 0;
  CUdeviceptr d_vertices = 0;
  CUdeviceptr d_indices = 0;

  OptixModule module = nullptr;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixPipeline pipeline = nullptr;


  OptixProgramGroup raygen_prog_group = nullptr;
  OptixProgramGroup miss_prog_groups[2];
  OptixProgramGroup hitgroup_prog_groups[2];

  CUstream stream = 0;
  PathTraceParams params = {};
  PathTraceParams* d_params = nullptr;

  OptixShaderBindingTable sbt = {};
};


struct OptixSettings {
  void (*logCallback)(unsigned int level, const char* tag, const char* message, void* cbdata) = nullptr;
  unsigned int logCallbackLevel = 4;
  #ifdef DEBUG_MODE
    OptixDeviceContextValidationMode validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    OptixCompileOptimizationLevel optimizationLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    OptixCompileDebugLevel debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  #endif // DEBUG_MODE
  #ifdef RELEASE_MODE
    OptixCompileOptimizationLevel optimizationLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    OptixCompileDebugLevel debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  #endif // RELEASE_MODE

  TinyObjWrapper scene;
  unsigned int scene_width = 800;
  unsigned int scene_height = 600;
  unsigned int samples_per_launch = 1;
  unsigned int max_depth = 5;
  bool useDirectLighting = true;
  bool useImportanceSampling = true;
  float* volume_data = nullptr;
  size_t volume_sz = 0;
  size_t volume_width = 0;
  size_t volume_height = 0;
  size_t volume_depth = 0;
};




class OptixManager
{
public:
  class UserInterface {
  public:
    UserInterface(OptixManager* manager) : _manager(manager) {}
    sutil::Camera g_camera;
    sutil::Trackball trackball;
    int32_t mouse_button = -1;
    uint16_t width = 800;
    uint16_t height = 600;
    bool camera_changed = true;
    bool refreshAccumulationBuffer = true;
    void init_camera();
    float3 translateCamera(float3 translation);
    void handleCameraUpdate();
  private:
    OptixManager* _manager = nullptr;
  } ui;

  class GLManager {
  public:
    bool init(OptixManager*);
    GLFWwindow* window = nullptr;
    std::shared_ptr <sutil::GLDisplay>  gl_display;
    bool showCurrentFrame(std::shared_ptr<sutil::CUDAOutputBuffer<uchar4>>);
    void pollEvents();
    bool shouldClose();
    void swapBuffers();
    void dispose();
  };

  OptixManager(const OptixSettings& settings);
  ~OptixManager();
  bool init();
  void render_loop();
  void dispose();
  void print_benchmark();

  void set_use_direct_lighting(bool useDirectLighting);
  void toggle_direct_lighting();
  void set_use_importance_sampling(bool useImportanceSampling);
  void toggle_importance_sampling();

  void increase_recursion_depth();
  void decrease_recurstion_depth();

  bool use_direct_lighting() const;
  bool use_importance_sampling() const;
  int max_depth() const;

  void set_camera_eye(float3 eye);
  void update_camera_UVW(sutil::Camera& g_camera);


private:
  OptixSettings _settings;
  PathTracerState _state;
  sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
  
  std::shared_ptr<sutil::CUDAOutputBuffer<uchar4>> _output_buffer;
  
  GLManager _glManager;



  bool initializeTheLaunch();
  bool updateState();
  bool launchCurrentFrame();
  bool createDeviceContext();
  bool buildTheAccelarationStructure();
  bool createModule();
  bool createProgramGroups();
  bool createPipeline();
  bool createShaderBindingTable();
  
  // performance counters
  uint32_t sample_sum = 0;
  uint32_t frame_count = 0;
  float frame_time = 0.0f;
  float total_time = 0.0f;





};
#endif // !OPTIX_MANAGER_H