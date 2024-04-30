#ifndef OPTIX_MANAGER_H
#define OPTIX_MANAGER_H

#include "common.h"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Trackball.h>
#include <sutil/Camera.h>
#include <sutil/GLDisplay.h>


#include "tinylogger.h"
#include "GLFCallbacks.h"
#include "pathTracer.h"



#include "TinyObjWrapper.h"


constexpr unsigned int NUM_RAYTYPES = 2; // 0: radiance, 1: volume

constexpr OptixPayloadTypeID RADIANCE_PAYLOAD_TYPE = OPTIX_PAYLOAD_TYPE_ID_0;
constexpr OptixPayloadTypeID VOLUME_PAYLOAD_TYPE = OPTIX_PAYLOAD_TYPE_ID_1;

enum DoneReason {
  MISS,
  MAX_DEPTH,
  RUSSIAN_ROULETTE,
  LIGHT_HIT,
  ABSORBED,
  VOLUME_BOUNDARY,
  NOT_DONE
};

struct RadiancePayloadRayData {
  float3 attenuation; // Accumulated ray color
  unsigned int randomSeed; // Current random seed used for Monte Carlo sampling
  int depth; // Current recursion depth
  float3 emissionColor; // Emission color of the current surface
  float3 radiance; // Accumulated radiance
  //float importance; // Importance of the current path TODO: Implement this
  float3 origin; // Origin of the current ray
  float3 direction; // Direction of the current ray
  int done; // Flag to indicate if the current path is done
  DoneReason doneReason; // Reason why the current path is done
}; // 19 parameters

struct VolumePayLoadRayData {
  float3 attenuation;         //  3 idx 0, 1, 2
  unsigned int randomSeed;    //  1 idx 3
  int step;                   //  1 idx 4
  float3 emissionColor;       //  3 idx 5, 6, 7
  float3 radiance;            //  3 idx 8, 9, 10
  float3 origin;              //  3 idx 11, 12, 13
  float3 direction;           //  3 idx 14, 15, 16
  int done;                   //  1 idx 17
  DoneReason doneReason;      //  1 idx 18
  float tMax;                 //  1 idx 19
  OptixAabb volumeAabb;       // +6 idx 20, 21, 22, 23, 24, 25
  //----
};                            // 26 parameters


const unsigned int radiancePayloadRayDataSemantics[19] =
{
  // RadiancePayloadRayData::attenuation
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
  // RadiancePayloadRayData::seed
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
  // RadiancePayloadRayData::depth
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
  // RadiancePayloadRayData::emitted
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
  // RadiancePayloadRayData::radiance
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
  // RadiancePayloadRayData::origin
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
  // RadiancePayloadRayData::direction
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
  // RadiancePayloadRayData::done
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
  // RadiancePayloadRayData::doneReason
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE
};

const unsigned int volumePayloadRayDataSemantics[26] =
{
  // VolumePayloadRayData::attenuation
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE, //0
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE, //1
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE, //2
  // VolumePayloadRayData::seed
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE, //3
  // VolumePayloadRayData::step
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE, //4
  // VolumePayloadRayData::emitted
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE, //5
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE, //6
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE, //7
  // VolumePayloadRayData::radiance
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE, //8
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE, //9
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE, //10
  // VolumePayloadRayData::origin
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE, //11
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE, //12
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE, //13
  // VolumePayloadRayData::direction
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE, //14
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE, //15
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE, //16
  // VolumePayloadRayData::done
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE, //17
  // VolumePayloadRayData::doneReason
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE, //18
  // VolumePayloadRayData::tmax
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE, //19
  // VolumePayloadRayData::volumeAabb -- minX, minY, minZ, maxX, maxY, maxZ
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ, //20
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ, //21
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ, //22
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ, //23
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ, //24
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ  //25
};


struct AreaLight {
  float3 corner;
  float3 v1;
  float3 v2;
  float3 normal;
  float3 emission;
};

struct PathTraceParams {

  unsigned int width;
  unsigned int height;
  unsigned int maxDepth;
  bool useDirectLighting;
  bool useImportanceSampling;

  unsigned int currentFrameIdx;
  float4* accumulationBuffer;
  uchar4* frameBuffer;

  unsigned int samplesPerPixel;

  float3 cameraEye;
  float3 cameraU;
  float3 cameraV;
  float3 cameraW;

  AreaLight areaLight;
  OptixTraversableHandle handle;


  float refractiveRoughness;
  float metallicRoughness;


};

struct RayGenerationData {

};

struct MissData {
  float4 backgroundColor;
};

struct HitGroupData {
  BSDFType bsdfType;
  OptixAabb aabb;
  float3 emissionColor;
  float3 diffuseColor;
  float IOR;
  float roughness;
  float metallic;
  float4* vertices;
  uint3* indices;
};

struct ShadingParams
{
  float3 normal;
  float3 hitpoint;
  float3 direction;
  float3 attenuation;
  unsigned int seed;
  float3 origin;
};

struct VolumeSample
{
  float4 rgba;
  float density;
};

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
    OptixCompileOptimizationLevel optimizationLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
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
    sutil::GLDisplay gl_display;
    bool showCurrentFrame(sutil::CUDAOutputBuffer<uchar4>&);
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
  PathTraceParams _params;
  sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
  
  sutil::CUDAOutputBuffer<uchar4> _output_buffer;
  
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