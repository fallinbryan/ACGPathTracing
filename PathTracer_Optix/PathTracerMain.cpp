#include <glad/glad.h>  // Needs to be included before gl_interop


#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>


#include <iostream>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include  <chrono>

#include "TinyObjWrapper.h"
#include "pathTracer.h"


/*TODO: Load these parameters either from a config file or command line.  Possbly even could make a UI to set them at runtime*/
constexpr unsigned int maxiumumRecursionDepth = 28;
constexpr int32_t samples_per_launch = 128;
UINT32 frame_counter;
UINT32 sample_summ;
UINT32 avg_ms;
UINT32 total_ms;


const std::string objfilepath = "C:\\Users\\falli\\Documents\\CornellBoxWithMonkey.obj";

bool refreshAccumulationBuffer = false;

sutil::Camera g_camera;



int32_t width = 512;
int32_t height = 512;

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
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    
    CUstream stream = 0;
    PathTraceParams params = {};
    PathTraceParams* d_params = nullptr;

    OptixShaderBindingTable sbt = {};
};

float3 tinyobjToFloat3(const tinyobj::real_t* v)
{
  return make_float3(v[0], v[1], v[2]);
}

static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{

  PathTraceParams* params = static_cast<PathTraceParams*>(glfwGetWindowUserPointer(window));

  if (action == GLFW_PRESS)
  {
    if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
    {
      glfwSetWindowShouldClose(window, true);
    }
    else if (key == GLFW_KEY_0) {

      params->useDirectLighting = !params->useDirectLighting;
      std::cout << std::endl << "Using Direct Lighting: " << (params->useDirectLighting ? "yes" : "no") << std::endl;
      refreshAccumulationBuffer = true;
    }
    else if (key == GLFW_KEY_1) {

      params->useImportanceSampling = !params->useImportanceSampling;
      std::cout << std::endl << "Using Importance Sampling: " << (params->useImportanceSampling ? "yes" : "no") << std::endl;
      refreshAccumulationBuffer = true;
    }
    else if (key == GLFW_KEY_UP) {
      params->maxDepth = std::min((int)maxiumumRecursionDepth, (int)params->maxDepth + 1);
      refreshAccumulationBuffer = true;
      std::cout << std::endl << "Max Depth: " << params->maxDepth << std::endl;
    }
    else if (key == GLFW_KEY_DOWN) {
      params->maxDepth = std::max(1, (int)params->maxDepth - 1);
      refreshAccumulationBuffer = true;
      std::cout << std::endl << "Max Depth: " << params->maxDepth << std::endl;
    }
    else if (key == GLFW_KEY_R) {
      refreshAccumulationBuffer = true;
    }
  }
  else if (key == GLFW_KEY_G)
  {
    // toggle UI draw
  }
}

void initializeTheLaunch(PathTracerState& state) { 
  
  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void**>(&state.params.accumulationBuffer),
    state.params.width * state.params.height * sizeof(float4)
  ));

  state.params.frameBuffer = nullptr;
  state.params.samplesPerPixel = samples_per_launch;
  state.params.currentFrameIdx = 0u;

  state.params.areaLight.emission = make_float3(10.0f, 10.0f, 10.0f);
  state.params.areaLight.corner = make_float3(343.0f, 547.0f, 227.0f);
  state.params.areaLight.v1 = make_float3(0.0f, 0.0f, 105.0f);
  state.params.areaLight.v2 = make_float3(-130.0f, 0.0f, 0.0f);
  state.params.areaLight.normal = normalize(cross(state.params.areaLight.v1, state.params.areaLight.v2));
  state.params.handle = state.gas_handle;

  CUDA_CHECK( cudaStreamCreate( &state.stream ) );
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( PathTraceParams ) ) );

}

void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, PathTracerState& state)
{
 
  if (refreshAccumulationBuffer)
  {
    refreshAccumulationBuffer = false;
    state.params.currentFrameIdx = 0;
    sample_summ = 0;
    frame_counter = 0;
    avg_ms = 0;
    total_ms = 0;

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accumulationBuffer)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.params.accumulationBuffer), state.params.width * state.params.height * sizeof(float4)));
  }
  
}

void LaunchCurrentFrame(sutil::CUDAOutputBuffer<uchar4>& output_buffer, PathTracerState& state) {

  uchar4* result_buffer_data = output_buffer.map();

  state.params.frameBuffer = result_buffer_data;


  CUDA_CHECK(cudaMemcpyAsync(
    reinterpret_cast<void*>(state.d_params),
    &state.params, sizeof(PathTraceParams),
    cudaMemcpyHostToDevice, state.stream
  ));

  OPTIX_CHECK(optixLaunch(
    state.pipeline,
    state.stream,
    reinterpret_cast<CUdeviceptr>( state.d_params ),
    sizeof( PathTraceParams ),
    &state.sbt,
    state.params.width,
    state.params.height,
    1
  ) );

  output_buffer.unmap();
  CUDA_SYNC_CHECK();
}

void showCurrentFrame(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window) {

  int frame_resolution_x = 0;
  int frame_resolution_y = 0;
  glfwGetFramebufferSize(window, &frame_resolution_x, &frame_resolution_y);

  gl_display.display(
    output_buffer.width(),
    output_buffer.height(),
    frame_resolution_x,
    frame_resolution_y,
    output_buffer.getPBO()
  );

}

void initCamera() {
  g_camera.setEye(make_float3(278.0f, 273.0f, -900.0f));
  g_camera.setLookat(make_float3(278.0f, 273.0f, 330.0f));
  g_camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
  g_camera.setFovY(35.0f);
}

static void context_log_callback(unsigned int level, const char* tag, const char* message, void* /*callbackdata */)
{
  std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

void createDeviceContext(PathTracerState& state) {

  CUDA_CHECK(cudaFree(0));

  OptixDeviceContext context;
  CUcontext cu_ctx = 0; 
  OPTIX_CHECK(optixInit());

  OptixDeviceContextOptions opts = {};
  opts.logCallbackFunction = context_log_callback;
  opts.logCallbackLevel = 4;
  opts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL; // remove this line for release builds

  OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &opts, &context));

  state.context = context;


}

void buildTheAccelarationStructure(PathTracerState& state, TinyObjWrapper objs) {
   // put all the geometry onto the device 
  //const size_t vetex_bytes_size = g_vertices.size() * sizeof(Vertex);

  std::vector<float> h_vertices = objs.getVerticesFloat();
  const size_t vetex_bytes_size = h_vertices.size() * sizeof(float);

  std::vector<uint32_t> h_mat_indices = objs.getMaterialIndices();
  const size_t mat_bytes_size = h_mat_indices.size() * sizeof(uint32_t);

  std::vector<uint32_t> h_indxbuffer = objs.getIndexBuffer();
  const size_t indx_bytes_size = h_indxbuffer.size() * sizeof(uint32_t);

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_vertices), vetex_bytes_size));
  CUDA_CHECK(cudaMemcpy(
    reinterpret_cast<void*>(state.d_vertices),
    h_vertices.data(),
    vetex_bytes_size,
    cudaMemcpyHostToDevice
  ));

  CUdeviceptr d_material_indices = 0;
  //const size_t mat_bytes_size = g_mat_indices.size() * sizeof(uint32_t);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_material_indices), mat_bytes_size));
  CUDA_CHECK(cudaMemcpy(
    reinterpret_cast<void*>(d_material_indices),
    h_mat_indices.data(),
    mat_bytes_size,
    cudaMemcpyHostToDevice
  ));

  CUdeviceptr d_index_buffer = 0;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_indices), indx_bytes_size));
  CUDA_CHECK(cudaMemcpy(
    reinterpret_cast<void*>(state.d_indices),
    h_indxbuffer.data(),
    indx_bytes_size,
    cudaMemcpyHostToDevice
  ));

  // BUILD THE BHV ( Optix calls it a GAS, but it's a BVH under the hood, so I'll call it a BHV ) 
  // Also, Optix uses the RT hardware to do the traversal so its super fast
  // This is the whole point of using Optix for this project because I have an RTX 3090 and I want to use it
  int mat_count = objs.getNumMaterials();

  std::unique_ptr<uint32_t[]> triangle_input_flags(new uint32_t[mat_count]);
  for (int i = 0; i < mat_count; i++) {
    triangle_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
  }


  OptixBuildInput triangle_input = {};
  triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangle_input.triangleArray.vertexStrideInBytes = sizeof(float) * 4;
  triangle_input.triangleArray.numVertices = static_cast<uint32_t>(h_vertices.size());
  triangle_input.triangleArray.vertexBuffers = &state.d_vertices;
  triangle_input.triangleArray.flags = triangle_input_flags.get();

  triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  triangle_input.triangleArray.indexStrideInBytes = 0;// sizeof(uint32_t);
  triangle_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(h_indxbuffer.size() / 3);
  triangle_input.triangleArray.indexBuffer = state.d_indices;

  triangle_input.triangleArray.numSbtRecords = mat_count;
  triangle_input.triangleArray.sbtIndexOffsetBuffer = d_material_indices;
  triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
  triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(
    state.context,
    &accel_options,
    &triangle_input,
    1,  // num_build_inputs
    &gas_buffer_sizes
  ));

  CUdeviceptr d_temp_buffer_gas;
  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void**>(&d_temp_buffer_gas),
    gas_buffer_sizes.tempSizeInBytes
  ));

  CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
  size_t compacted_gas_size_offset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
    compacted_gas_size_offset + 8
  ));

  OptixAccelEmitDesc emitProperty = {};
  emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compacted_gas_size_offset);

  OPTIX_CHECK(optixAccelBuild(
    state.context,
    0,  // CUDA stream
    &accel_options,
    &triangle_input,
    1,  // num build inputs
    d_temp_buffer_gas,
    gas_buffer_sizes.tempSizeInBytes,
    d_buffer_temp_output_gas_and_compacted_size,
    gas_buffer_sizes.outputSizeInBytes,
    &state.gas_handle,
    &emitProperty,  // emitted property list
    1   // num emitted properties
  ));

  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_material_indices)));

  size_t compacted_gas_size;
  CUDA_CHECK(cudaMemcpy(&compacted_gas_size, reinterpret_cast<void*>(emitProperty.result), sizeof(size_t), cudaMemcpyDeviceToHost));

  if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer), compacted_gas_size));

    OPTIX_CHECK(optixAccelCompact(
      state.context,
      0,  // CUDA stream
      state.gas_handle,
      state.d_gas_output_buffer,
      compacted_gas_size,
      &state.gas_handle
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_buffer_temp_output_gas_and_compacted_size)));
  }
  else {
    state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
  
  }
}

void createModule(PathTracerState& state) {
 
  OptixPayloadType payloadType = {};

  payloadType.numPayloadValues = sizeof(radiancePayloadRayDataSemantics) / sizeof(radiancePayloadRayDataSemantics[0]);
  payloadType.payloadSemantics = radiancePayloadRayDataSemantics;

  OptixModuleCompileOptions module_compile_options = {};

  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;/// OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // TODO: Change this to 3 for release builds
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE; // OPTIX_COMPILE_DEBUG_LEVEL_FULL;  // TODO: Change this to 0 for release builds

  module_compile_options.numPayloadTypes = 1; // TODO: Change this to 2 for shadow rays
  module_compile_options.payloadTypes = &payloadType;

  state.pipeline_compile_options.usesMotionBlur = false;
  state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  state.pipeline_compile_options.numPayloadValues = 0;
  state.pipeline_compile_options.numAttributeValues = 2;
  state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

  size_t inputSize = 0;
  const char* ptx = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "pathTracerPrograms.cu", inputSize );

  OPTIX_CHECK_LOG(optixModuleCreate(
    state.context,
    &module_compile_options,
    &state.pipeline_compile_options,
    ptx,
    inputSize,
    LOG, &LOG_SIZE,
    &state.module
  ));

}


void createProgramGroups(PathTracerState& state) {
  OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
  {


    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = state.module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
      state.context,
      &raygen_prog_group_desc,
      1,  // num program groups
      &program_group_options,
      LOG, &LOG_SIZE,
      &state.raygen_prog_group
    ));
  }

  {
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = state.module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
      state.context,
      &miss_prog_group_desc,
      1,  // num program groups
      &program_group_options,
      LOG, &LOG_SIZE,
      &state.miss_prog_group
    ));
  }

  {
    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__diffuse__ch";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
      state.context,
      &hitgroup_prog_group_desc,
      1,  // num program groups
      &program_group_options,
      LOG, &LOG_SIZE,
      &state.hitgroup_prog_group
    ));
  
  }
}

void createPipeline(PathTracerState& state) {
  OptixProgramGroup program_groups[] = { state.raygen_prog_group, state.miss_prog_group, state.hitgroup_prog_group };
  

  OptixPipelineLinkOptions pipeline_link_options = {};
  pipeline_link_options.maxTraceDepth = maxiumumRecursionDepth;


  OPTIX_CHECK_LOG(optixPipelineCreate(
    state.context,
    &state.pipeline_compile_options,
    &pipeline_link_options,
    program_groups,
    sizeof(program_groups) / sizeof(program_groups[0]),
    LOG, &LOG_SIZE,
    &state.pipeline
  ));

  OptixStackSizes stack_sizes = {};
  OPTIX_CHECK( optixUtilAccumulateStackSizes( state.raygen_prog_group, &stack_sizes, state.pipeline ) );
  OPTIX_CHECK( optixUtilAccumulateStackSizes( state.miss_prog_group, &stack_sizes, state.pipeline ) );
  OPTIX_CHECK( optixUtilAccumulateStackSizes( state.hitgroup_prog_group, &stack_sizes, state.pipeline ) );

  uint32_t max_trace_depth = maxiumumRecursionDepth; 
  uint32_t max_cc_depth = 0;
  uint32_t max_dc_depth = 0;
  uint32_t direct_callable_stack_size_from_traversal;
  uint32_t direct_callable_stack_size_from_state;
  uint32_t continuation_stack_size;
  OPTIX_CHECK( optixUtilComputeStackSizes( 
    &stack_sizes, 
    max_trace_depth, 
    max_cc_depth, 
    max_dc_depth, 
    &direct_callable_stack_size_from_traversal, 
    &direct_callable_stack_size_from_state,
    &continuation_stack_size
  ) );

  const uint32_t max_traversal_depth = 1;
  OPTIX_CHECK(optixPipelineSetStackSize(
    state.pipeline, 
    direct_callable_stack_size_from_traversal, 
    direct_callable_stack_size_from_state, 
    continuation_stack_size, 
    max_traversal_depth
  ) );

}




void createShaderBindingTable(PathTracerState& state, TinyObjWrapper obj) {

  std::vector<Material> materials = obj.getMaterials();

  CUdeviceptr d_raygen_record;
  const size_t raygen_record_size = sizeof(RayGenerationRecord);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

  RayGenerationRecord raygen_record = {};
  OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &raygen_record));

  CUDA_CHECK(cudaMemcpy(
    reinterpret_cast<void*>(d_raygen_record),
    &raygen_record,
    raygen_record_size,
    cudaMemcpyHostToDevice
  ));

  CUdeviceptr d_miss_record;
  const size_t miss_record_size = sizeof(MissRecord);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), miss_record_size * NUM_RAYTYPES));

  MissRecord miss_record[1];
  OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &miss_record[0] ));
  miss_record[0].data.backgroundColor = make_float4(0.0f);

  CUDA_CHECK(cudaMemcpy(
    reinterpret_cast<void*>(d_miss_record),
    miss_record,
    miss_record_size * NUM_RAYTYPES,
    cudaMemcpyHostToDevice
  ));

  CUdeviceptr d_hitgroup_record;
  const size_t hitgroup_record_size = sizeof(HitGroupRecord);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), hitgroup_record_size * NUM_RAYTYPES * materials.size()));

  
  //std::array<HitGroupRecord,h_diffuse_colors.size() * NUM_RAYTYPES> hitgroup_records;

  size_t mat_count = materials.size();

  HitGroupRecord* hitgroup_records = new HitGroupRecord[mat_count * NUM_RAYTYPES];
  for (int i = 0; i < materials.size(); i++)
  {

    {
      const int shaderBindingTableIndex = i * NUM_RAYTYPES + 0;

      

      OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_prog_group, &hitgroup_records[shaderBindingTableIndex]));

      

      hitgroup_records[shaderBindingTableIndex].data.emissionColor = materials[i].emission;
      hitgroup_records[shaderBindingTableIndex].data.diffuseColor = materials[i].diffuse;
      hitgroup_records[shaderBindingTableIndex].data.bsdfType = materials[i].bsdfType;
      hitgroup_records[shaderBindingTableIndex].data.roughness = materials[i].roughness;
      hitgroup_records[shaderBindingTableIndex].data.IOR = materials[i].ior;
      hitgroup_records[shaderBindingTableIndex].data.metallic = materials[i].metallic;
      hitgroup_records[shaderBindingTableIndex].data.vertices = reinterpret_cast<float4*>(state.d_vertices);
      hitgroup_records[shaderBindingTableIndex].data.indices = reinterpret_cast<uint3*>(state.d_indices);
    }
  }

  CUDA_CHECK(cudaMemcpy(
    reinterpret_cast<void*>(d_hitgroup_record),
    hitgroup_records,
    hitgroup_record_size * NUM_RAYTYPES * mat_count,
    cudaMemcpyHostToDevice
  ));

  state.sbt.raygenRecord = d_raygen_record;
  state.sbt.missRecordBase = d_miss_record;
  state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
  state.sbt.missRecordCount = NUM_RAYTYPES;
  state.sbt.hitgroupRecordBase = d_hitgroup_record;
  state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
  state.sbt.hitgroupRecordCount = NUM_RAYTYPES * mat_count;

  delete[] hitgroup_records;

}

void CleanAllTheThings(PathTracerState& state) {
  OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
  OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
  OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
  OPTIX_CHECK(optixProgramGroupDestroy(state.hitgroup_prog_group));
  OPTIX_CHECK(optixModuleDestroy(state.module));
  OPTIX_CHECK(optixDeviceContextDestroy(state.context));

  CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord ) ) );
  CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) );
  CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
  CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_vertices )));
  CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer )));
  CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.accumulationBuffer )));
  CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params )));


} 

void main() {
    
    TinyObjWrapper obj(objfilepath);

    PathTracerState state;
    state.params.width = width;
    state.params.height = height;
    state.params.useDirectLighting = false;
    state.params.useImportanceSampling = false;
    state.params.maxDepth = 4;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    try {

      initCamera();

      g_camera.setAspectRatio(static_cast<float>(state.params.width) / static_cast<float>(state.params.height));
      state.params.cameraEye = g_camera.eye();
      g_camera.UVWFrame(state.params.cameraU, state.params.cameraV, state.params.cameraW);


      std::cout << "Using Direct Lighting: " << (state.params.useDirectLighting ? "yes" : "no") << std::endl;
      std::cout << "Using Importance Sampling: " << (state.params.useImportanceSampling ? "yes" : "no") << std::endl;

      createDeviceContext(state);
      buildTheAccelarationStructure(state, obj);
      std::cout << "Acceleration Structure Built" << std::endl;
      createModule(state);
      std::cout << "Module Created" << std::endl;
      createProgramGroups(state);
      std::cout << "Program Groups Created" << std::endl;
      createPipeline(state);
      std::cout << "Pipeline Created" << std::endl;
      createShaderBindingTable(state, obj);
      std::cout << "Shader Binding Table Created" << std::endl;
      initializeTheLaunch(state);
      std::cout << "Launch Initialized" << std::endl;

      GLFWwindow* window = sutil::initUI( "Path Tracer", state.params.width, state.params.height );
      glfwSetWindowUserPointer(window, &state.params);
      glfwSetKeyCallback(window, keyCallback);

      {
        sutil::CUDAOutputBuffer<uchar4> output_buffer(
          output_buffer_type, 
          state.params.width, 
          state.params.height
        );

        output_buffer.setStream(state.stream);
        sutil::GLDisplay gl_display;
        
       do
       {

         auto start = std::chrono::high_resolution_clock::now();

          glfwPollEvents();
          updateState(output_buffer, state);

          LaunchCurrentFrame(output_buffer, state);

          showCurrentFrame(output_buffer, gl_display, window);

          glfwSwapBuffers(window);

          ++state.params.currentFrameIdx;

          auto end = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

          avg_ms += duration.count();
          total_ms += duration.count();


          sample_summ += samples_per_launch;
          frame_counter++;

          std::cout << "\rFrame Render Time: " << duration.count() << "ms" << std::flush;
   
          

       } while (!glfwWindowShouldClose(window));
        CUDA_SYNC_CHECK();
      }
      sutil::cleanupUI(window);
      CleanAllTheThings(state);
      if (frame_counter > 0)
        avg_ms /= frame_counter;

      std::cout << "Total Samples " << sample_summ << std::endl;
      std::cout << "Average ms per frame: " << avg_ms << std::endl;
      std::cout << "Total ms: " << total_ms << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    
}