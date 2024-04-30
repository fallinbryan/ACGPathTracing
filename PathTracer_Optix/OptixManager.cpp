#include "common.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/GLDisplay.h>


#include "OptixManager.h"
#include "tinylogger.h"
#include "GLFCallbacks.h"
#include "pathTracer.h"



#define LOG_ERROR(x) tlog::error(x)

#pragma region Helper Functions

static OptixSettings copySettings(const OptixSettings& src)
{
  OptixSettings dst;
  dst.logCallback = src.logCallback;
  dst.logCallbackLevel = src.logCallbackLevel;
#ifdef DEBUG_MODE
  dst.validationMode = src.validationMode;
#endif // DEBUG_MODE
  dst.scene = src.scene;
  dst.scene_width = src.scene_width;
  dst.scene_height = src.scene_height;
  dst.samples_per_launch = src.samples_per_launch;
  dst.max_depth = src.max_depth;
  dst.useDirectLighting = src.useDirectLighting;
  dst.useImportanceSampling = src.useImportanceSampling;
  return dst;
}

#pragma endregion

#pragma region OptixManager Public Functions

  OptixManager::OptixManager(const OptixSettings& settings) :
    _settings(copySettings(settings)),
    ui(this)
  {
    // Constructor body
    _state.params.width = settings.scene_width;
    _state.params.height = settings.scene_height;
    _state.params.maxDepth = settings.max_depth;
    _state.params.useDirectLighting = true;
    _state.params.useImportanceSampling = true;

    ui.width = settings.scene_width;
    ui.height = settings.scene_height;
    
  }

  OptixManager::~OptixManager() {
    dispose();
  }

  bool OptixManager::init()
  {
    ui.init_camera();

    if (!createDeviceContext())
    {
      LOG_ERROR("Failed to initialize Optix");
      return false;
    }

    if (!buildTheAccelarationStructure())
    {
      LOG_ERROR("Failed to build the GAS");
      return false;
    }

    if (!createModule())
    {
      LOG_ERROR("Failed to create Optix Module");
      return false;
    }

    if (!createProgramGroups())
    {
      LOG_ERROR("Failed create Program Groups");
      return false;
    }

    if (!createPipeline())
    {
      LOG_ERROR("Failed to create the OptixPipeline");
      return false;
    }

    if (!createShaderBindingTable())
    {
      LOG_ERROR("Failed to create SBT");
      return false;
    }

    if (!initializeTheLaunch())
    {
      LOG_ERROR("Failed to initialize the launch");
      return false;
    }

    if (!_glManager.init(this))
    {
      LOG_ERROR("Failed to initialize OpenGL");
      return false;
    } 
    _output_buffer = std::make_shared<sutil::CUDAOutputBuffer<uchar4>>(output_buffer_type, ui.width, ui.height);

    _output_buffer->setStream(_state.stream);

    return true;
  }

  void OptixManager::set_camera_eye(float3 eye)
  {
    _state.params.cameraEye = eye;
  }

  void OptixManager::update_camera_UVW(sutil::Camera& g_camera)
  {
    g_camera.UVWFrame(_state.params.cameraU, _state.params.cameraV, _state.params.cameraW);
  }

  void OptixManager::increase_recursion_depth()
  {

    _state.params.maxDepth = std::max(1, (int)_settings.max_depth - 1);
  }

  void OptixManager::decrease_recurstion_depth()
  {
    _state.params.maxDepth = std::min((int)_settings.max_depth, (int)_state.params.maxDepth + 1);
  }

  void OptixManager::dispose()
  {
    OptixProgramGroup program_groups[] = {
        _state.raygen_prog_group,
        _state.miss_prog_groups[0],
        _state.miss_prog_groups[1],
        _state.hitgroup_prog_groups[0],
        _state.hitgroup_prog_groups[1]
    };
    OPTIX_CHECK(optixPipelineDestroy(_state.pipeline));

    for (auto& pg : program_groups)
      OPTIX_CHECK(optixProgramGroupDestroy(pg));

    OPTIX_CHECK(optixModuleDestroy(_state.module));
    OPTIX_CHECK(optixDeviceContextDestroy(_state.context));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state.d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state.d_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state.params.accumulationBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state.d_params)));

  }

  void OptixManager::render_loop() 
  {
    do {

      auto start = std::chrono::high_resolution_clock::now();

      _glManager.pollEvents();
      updateState();
      launchCurrentFrame();
      _glManager.showCurrentFrame(_output_buffer);
      _glManager.swapBuffers();
      ++_state.params.currentFrameIdx;

      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

      frame_time = duration.count();
      total_time += frame_time;
      sample_sum += _settings.samples_per_launch;
      frame_count++;

      std::cout << "\rFrame Render Time: " << frame_time << "ms" << std::flush;

    } while (!_glManager.shouldClose());
    CUDA_SYNC_CHECK();
  }

  void OptixManager::print_benchmark()
  {
    if (frame_count > 0)
    {
      float avg_ms = total_time / frame_count;

      tlog::info() << "Total Samples " << sample_sum;
      tlog::info() << "Average ms per frame: " << avg_ms;
      tlog::info() << "Total ms: " << total_time;

    }
  }

  void OptixManager::set_use_direct_lighting(bool useDirectLighting)
  {
    _state.params.useDirectLighting = useDirectLighting;
  }

  void OptixManager::toggle_direct_lighting()
  {
    _state.params.useDirectLighting = !_state.params.useDirectLighting;
  }

  void OptixManager::set_use_importance_sampling(bool useImportanceSampling)
  {
    _state.params.useImportanceSampling = useImportanceSampling;
  }

  void OptixManager::toggle_importance_sampling()
  {
    _state.params.useImportanceSampling = !_state.params.useImportanceSampling;
  }

  bool OptixManager::use_direct_lighting() const
  {
    return _state.params.useDirectLighting;
  }

  bool OptixManager::use_importance_sampling() const
  {
    return _state.params.useImportanceSampling;
  }

  int OptixManager::max_depth() const
  {
    return _state.params.maxDepth;
  }

#pragma endregion

#pragma region OptixManager Private Functions

  bool OptixManager::createDeviceContext()
  {
    try {

      CUDA_CHECK(cudaFree(0));

      OptixDeviceContext context;
      CUcontext cu_ctx = 0;
      OPTIX_CHECK(optixInit());

      OptixDeviceContextOptions opts = {};
      opts.logCallbackFunction = _settings.logCallback;
      opts.logCallbackLevel = _settings.logCallbackLevel;
#ifdef DEBUG_MODE
      opts.validationMode = _settings.validationMode;
#endif


      OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &opts, &context));

      _state.context = context;
    }
    catch (const std::exception& e)
    {
      LOG_ERROR(e.what());
      return false;
    }
    tlog::success() << "Optix Device Context Created";
    return true;
  }

  bool OptixManager::buildTheAccelarationStructure()
  {
    try
    {
      std::vector<float> h_vertices = _settings.scene.getVerticesFloat();
      const size_t vetex_bytes_size = h_vertices.size() * sizeof(float);

      std::vector<uint32_t> h_mat_indices = _settings.scene.getMaterialIndices();
      const size_t mat_bytes_size = h_mat_indices.size() * sizeof(uint32_t);

      std::vector<uint32_t> h_indxbuffer = _settings.scene.getIndexBuffer();
      const size_t indx_bytes_size = h_indxbuffer.size() * sizeof(uint32_t);

      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_state.d_vertices), vetex_bytes_size));
      CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(_state.d_vertices),
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
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_state.d_indices), indx_bytes_size));
      CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(_state.d_indices),
        h_indxbuffer.data(),
        indx_bytes_size,
        cudaMemcpyHostToDevice
      ));

      // BUILD THE BHV ( Optix calls it a GAS, but it's a BVH under the hood, so I'll call it a BHV ) 
      // Also, Optix uses the RT hardware to do the traversal so its super fast
      // This is the whole point of using Optix for this project because I have an RTX 3090 and I want to use it
      int mat_count = _settings.scene.getNumMaterials();

      std::unique_ptr<uint32_t[]> triangle_input_flags(new uint32_t[mat_count]);
      for (int i = 0; i < mat_count; i++) {
        triangle_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
      }


      OptixBuildInput triangle_input = {};
      triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      triangle_input.triangleArray.vertexStrideInBytes = sizeof(float) * 4;
      triangle_input.triangleArray.numVertices = static_cast<uint32_t>(h_vertices.size());
      triangle_input.triangleArray.vertexBuffers = &_state.d_vertices;
      triangle_input.triangleArray.flags = triangle_input_flags.get();

      triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      triangle_input.triangleArray.indexStrideInBytes = 0;// sizeof(uint32_t);
      triangle_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(h_indxbuffer.size() / 3);
      triangle_input.triangleArray.indexBuffer = _state.d_indices;

      triangle_input.triangleArray.numSbtRecords = mat_count;
      triangle_input.triangleArray.sbtIndexOffsetBuffer = d_material_indices;
      triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
      triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

      OptixAccelBuildOptions accel_options = {};
      accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
      accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

      OptixAccelBufferSizes gas_buffer_sizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage(
        _state.context,
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
        _state.context,
        0,  // CUDA stream
        &accel_options,
        &triangle_input,
        1,  // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &_state.gas_handle,
        &emitProperty,  // emitted property list
        1   // num emitted properties
      ));

      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_material_indices)));

      size_t compacted_gas_size;
      CUDA_CHECK(cudaMemcpy(&compacted_gas_size, reinterpret_cast<void*>(emitProperty.result), sizeof(size_t), cudaMemcpyDeviceToHost));

      if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_state.d_gas_output_buffer), compacted_gas_size));

        OPTIX_CHECK(optixAccelCompact(
          _state.context,
          0,  // CUDA stream
          _state.gas_handle,
          _state.d_gas_output_buffer,
          compacted_gas_size,
          &_state.gas_handle
        ));

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_buffer_temp_output_gas_and_compacted_size)));
      }
      else {
        _state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;

      }
    }
    catch (const std::exception& e)
    {
      LOG_ERROR(e.what());
      return false;
    }
    tlog::success() << "Optix GAS Created";
    return true;
  }

  bool OptixManager::createModule()
  {
    try
    {
      OptixPayloadType radiancePayloadType = {};
      radiancePayloadType.numPayloadValues = sizeof(radiancePayloadRayDataSemantics) / sizeof(radiancePayloadRayDataSemantics[0]);
      radiancePayloadType.payloadSemantics = radiancePayloadRayDataSemantics;

      OptixPayloadType volumePayloadType = {};
      volumePayloadType.numPayloadValues = sizeof(volumePayloadRayDataSemantics) / sizeof(volumePayloadRayDataSemantics[0]);
      volumePayloadType.payloadSemantics = volumePayloadRayDataSemantics;

      OptixPayloadType payloadTypes[] = { radiancePayloadType, volumePayloadType };

      OptixModuleCompileOptions module_compile_options = {};

      module_compile_options.optLevel = _settings.optimizationLevel;
      module_compile_options.debugLevel = _settings.debugLevel;// OPTIX_COMPILE_DEBUG_LEVEL_FULL;  // TODO: Change this to OPTIX_COMPILE_DEBUG_LEVEL_NONE for release builds

      module_compile_options.numPayloadTypes = 2;
      module_compile_options.payloadTypes = payloadTypes;

      _state.pipeline_compile_options.usesMotionBlur = false;
      _state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
      _state.pipeline_compile_options.numPayloadValues = 0;
      _state.pipeline_compile_options.numAttributeValues = 2;
      _state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
      _state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

      size_t inputSize = 0;
      const char* ptx = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "pathTracerPrograms.cu", inputSize);

      OPTIX_CHECK_LOG(optixModuleCreate(
        _state.context,
        &module_compile_options,
        &_state.pipeline_compile_options,
        ptx,
        inputSize,
        LOG, &LOG_SIZE,
        &_state.module
      ));
    }
    catch (const std::exception& e)
    {
      LOG_ERROR(e.what());
      return false;
    }
    tlog::success() << "Optix Module Created";
    return true;
  }

  bool OptixManager::createProgramGroups()
  {
    try
    {
      OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
      {


        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = _state.module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
          _state.context,
          &raygen_prog_group_desc,
          1,  // num program groups
          &program_group_options,
          LOG, &LOG_SIZE,
          &_state.raygen_prog_group
        ));
      }

      {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = _state.module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
          _state.context,
          &miss_prog_group_desc,
          1,  // num program groups
          &program_group_options,
          LOG, &LOG_SIZE,
          &_state.miss_prog_groups[0]
        ));
      }

      {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = _state.module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__volume__ms";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
          _state.context,
          &miss_prog_group_desc,
          1,  // num program groups
          &program_group_options,
          LOG, &LOG_SIZE,
          &_state.miss_prog_groups[1]
        ));
      }

      {
        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = _state.module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__diffuse__ch";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
          _state.context,
          &hitgroup_prog_group_desc,
          1,  // num program groups
          &program_group_options,
          LOG, &LOG_SIZE,
          &_state.hitgroup_prog_groups[0]
        ));
      }

      {
        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = _state.module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__volume__ch";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
          _state.context,
          &hitgroup_prog_group_desc,
          1,  // num program groups
          &program_group_options,
          LOG, &LOG_SIZE,
          &_state.hitgroup_prog_groups[1]
        ));
      }
    }
    catch (const std::exception& e)
    {
      LOG_ERROR(e.what());
      return false;
    }
    tlog::success() << "Optix Program Groups Created";
    return true;
  }

  bool OptixManager::createPipeline()
  {
    try
    {
      OptixProgramGroup program_groups[] = {
        _state.raygen_prog_group,
        _state.miss_prog_groups[0],
        _state.miss_prog_groups[1],
        _state.hitgroup_prog_groups[0],
        _state.hitgroup_prog_groups[1]
      };


      OptixPipelineLinkOptions pipeline_link_options = {};
      pipeline_link_options.maxTraceDepth = 16;


      OPTIX_CHECK_LOG(optixPipelineCreate(
        _state.context,
        &_state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        LOG, &LOG_SIZE,
        &_state.pipeline
      ));

      OptixStackSizes stack_sizes = {};

      for (auto& pg : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, _state.pipeline));
      }


      uint32_t max_trace_depth = 16;
      uint32_t max_cc_depth = 0;
      uint32_t max_dc_depth = 0;
      uint32_t direct_callable_stack_size_from_traversal;
      uint32_t direct_callable_stack_size_from_state;
      uint32_t continuation_stack_size;
      OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
      ));

      const uint32_t max_traversal_depth = 1;
      OPTIX_CHECK(optixPipelineSetStackSize(
        _state.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
      ));

    }
    catch (const std::exception& e)
    {
      LOG_ERROR(e.what());
      return false;
    }
    tlog::success() << "Optix Pipeline Created";
    return true;
  }

  bool OptixManager::createShaderBindingTable()
  {
    try
    {
      std::vector<Material> materials = _settings.scene.getMaterials();
      std::vector<OptixAabb> aabbs = _settings.scene.getAabbs();

      CUdeviceptr d_raygen_record;
      const size_t raygen_record_size = sizeof(RayGenerationRecord);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

      RayGenerationRecord raygen_record = {};
      OPTIX_CHECK(optixSbtRecordPackHeader(_state.raygen_prog_group, &raygen_record));

      CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &raygen_record,
        raygen_record_size,
        cudaMemcpyHostToDevice
      ));

      /// MISS RECORDS

      CUdeviceptr d_miss_record;
      const size_t miss_record_size = sizeof(MissRecord);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), miss_record_size * NUM_RAYTYPES));

      MissRecord miss_record[NUM_RAYTYPES];

      for (int i = 0; i < NUM_RAYTYPES; i++) {
        OPTIX_CHECK(optixSbtRecordPackHeader(_state.miss_prog_groups[i], &miss_record[i]));
        miss_record[i].data.backgroundColor = make_float4(0.0f);
      }

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



      for (int j = 0; j < NUM_RAYTYPES; j++) {

        for (int i = 0; i < materials.size(); i++)
        {

          {
            const int shaderBindingTableIndex = i * NUM_RAYTYPES + j;
            OPTIX_CHECK(optixSbtRecordPackHeader(_state.hitgroup_prog_groups[j], &hitgroup_records[shaderBindingTableIndex]));


            hitgroup_records[shaderBindingTableIndex].data.emissionColor = materials[i].emission;
            hitgroup_records[shaderBindingTableIndex].data.diffuseColor = materials[i].diffuse;
            hitgroup_records[shaderBindingTableIndex].data.bsdfType = materials[i].bsdfType;
            hitgroup_records[shaderBindingTableIndex].data.roughness = materials[i].roughness;
            hitgroup_records[shaderBindingTableIndex].data.IOR = materials[i].ior;
            hitgroup_records[shaderBindingTableIndex].data.metallic = materials[i].metallic;
            hitgroup_records[shaderBindingTableIndex].data.vertices = reinterpret_cast<float4*>(_state.d_vertices);
            hitgroup_records[shaderBindingTableIndex].data.indices = reinterpret_cast<uint3*>(_state.d_indices);
            hitgroup_records[shaderBindingTableIndex].data.aabb = aabbs[10]; // <-- This is HACK, I need to fix this.  I need to get the AABB for the current object, except this block is associating materials with triangles, the pipeline has no concept of objects. I need to using the sbt index to get the correct AABB for the current object.


          }
        }
      }


      CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_record),
        hitgroup_records,
        hitgroup_record_size * NUM_RAYTYPES * mat_count,
        cudaMemcpyHostToDevice
      ));

      _state.sbt.raygenRecord = d_raygen_record;
      _state.sbt.missRecordBase = d_miss_record;
      _state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
      _state.sbt.missRecordCount = NUM_RAYTYPES;
      _state.sbt.hitgroupRecordBase = d_hitgroup_record;
      _state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
      _state.sbt.hitgroupRecordCount = NUM_RAYTYPES * mat_count;

      delete[] hitgroup_records;

    }
    catch (const std::exception& e)
    {
      LOG_ERROR(e.what());
      return false;
    }
    tlog::success() << "Optix Shader Binding Table Created";
    return true;
  }

  bool OptixManager::initializeTheLaunch()
  {
    try
    {
      CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&_state.params.accumulationBuffer),
        _state.params.width * _state.params.height * sizeof(float4)
      ));

      _state.params.frameBuffer = nullptr;
      _state.params.samplesPerPixel = _settings.samples_per_launch;
      _state.params.currentFrameIdx = 0u;

      _state.params.areaLight.emission = make_float3(10.0f, 10.0f, 10.0f);
      _state.params.areaLight.corner = make_float3(343.0f, 547.0f, 227.0f);
      _state.params.areaLight.v1 = make_float3(0.0f, 0.0f, 105.0f);
      _state.params.areaLight.v2 = make_float3(-130.0f, 0.0f, 0.0f);
      _state.params.areaLight.normal = normalize(cross(_state.params.areaLight.v1, _state.params.areaLight.v2));
      _state.params.handle = _state.gas_handle;
      _state.params.refractiveRoughness = 0.0f;
      _state.params.metallicRoughness = 0.0f;

      CUDA_CHECK(cudaStreamCreate(&_state.stream));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_state.d_params), sizeof(PathTraceParams)));

    }
    catch (const std::exception& e)
    {
      LOG_ERROR(e.what());
      return false;
    }
    tlog::success() << "Optix Launch Initialized";
    return true;
  }

  bool OptixManager::updateState()
  {
    try
    {
      if (ui.camera_changed) {
        ui.handleCameraUpdate();
      }

      if (ui.refreshAccumulationBuffer)
      {
        ui.refreshAccumulationBuffer = false;
        _state.params.currentFrameIdx = 0;
        sample_sum = 0;
        frame_count = 0;
        frame_time = 0;
        total_time = 0;

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state.params.accumulationBuffer)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_state.params.accumulationBuffer), _state.params.width * _state.params.height * sizeof(float4)));
      }
    }
    catch (const std::exception& e)
    {
      LOG_ERROR(e.what());
      return false;
    }
    
    return true;
  }

  bool OptixManager::launchCurrentFrame()
  {
    try
    {
      uchar4* result_buffer_data = _output_buffer->map();

      _state.params.frameBuffer = result_buffer_data;


      CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(_state.d_params),
        &_state.params, sizeof(PathTraceParams),
        cudaMemcpyHostToDevice, _state.stream
      ));

      OPTIX_CHECK(optixLaunch(
        _state.pipeline,
        _state.stream,
        reinterpret_cast<CUdeviceptr>(_state.d_params),
        sizeof(PathTraceParams),
        &_state.sbt,
        _state.params.width,
        _state.params.height,
        1
      ));

      _output_buffer->unmap();
      CUDA_SYNC_CHECK();
    }
    catch (const std::exception& e)
    {
      LOG_ERROR(e.what());
      cudaError_t error = cudaGetLastError();
      if (error != cudaSuccess) {
        
        LOG_ERROR(cudaGetErrorString(error));
      }

      throw e;
    }
    
    return true;
  }

  

#pragma endregion

#pragma region OptixManager::UserInterface Public Functions

  void OptixManager::UserInterface::init_camera()
  {
    // Initialize the camera
    g_camera.setEye(make_float3(278.0f, 273.0f, -900.0f));
    g_camera.setLookat(make_float3(278.0f, 273.0f, 330.0f));
    g_camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    g_camera.setFovY(35.0f);

    trackball.setCamera(&g_camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(
      make_float3(1.0f, 0.0f, 0.0f),
      make_float3(0.0f, 0.0f, 1.0f),
      make_float3(0.0f, 1.0f, 0.0f)
    );

    trackball.setGimbalLock(true);

    g_camera.setAspectRatio((float)width / (float)height);
    _manager->set_camera_eye(g_camera.eye());
    _manager->update_camera_UVW(g_camera);
  }

  float3 OptixManager::UserInterface::translateCamera(float3 translation)
  {
    // Translate the camera
    g_camera.setEye(g_camera.eye() + translation);
    _manager->set_camera_eye(g_camera.eye());
    return g_camera.eye();
  }

  void OptixManager::UserInterface::handleCameraUpdate()
  {
    if(!camera_changed)
      return;
    camera_changed = false;

    g_camera.setAspectRatio((float)width / (float)height);
    _manager->set_camera_eye(g_camera.eye());
    _manager->update_camera_UVW(g_camera);
    refreshAccumulationBuffer = true;
  }

#pragma endregion

#pragma region OptixManager::GLManager Public Functions


  bool OptixManager::GLManager::init(OptixManager* manager)
{
  try {

    window = sutil::initUI("Path Tracer", manager->ui.width , manager->ui.height);
    glfwSetWindowUserPointer(window, manager);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);

    gl_display = std::make_shared<sutil::GLDisplay>();

  }
  catch (const std::exception& e)
  {
    LOG_ERROR(e.what());
    return false;
  }

 
  return true;
}

  bool OptixManager::GLManager::showCurrentFrame(std::shared_ptr<sutil::CUDAOutputBuffer<uchar4>> output_buffer)
  {
    try
    {
      int frame_resolution_x = 0;
      int frame_resolution_y = 0;
      glfwGetFramebufferSize(window, &frame_resolution_x, &frame_resolution_y);

      gl_display->display(
        output_buffer->width(),
        output_buffer->height(),
        frame_resolution_x,
        frame_resolution_y,
        output_buffer->getPBO()
      );
    }
    catch (const std::exception& e)
    {
      LOG_ERROR(e.what());
      return false;
    }
    return true;
  }

  void OptixManager::GLManager::pollEvents()
  {
    glfwPollEvents();
  }

  void OptixManager::GLManager::swapBuffers()
  {
    glfwSwapBuffers(window);
  }

  bool OptixManager::GLManager::shouldClose()
  {
    return glfwWindowShouldClose(window);
  }

  void OptixManager::GLManager::dispose()
  {
    sutil::cleanupUI(window);
  }

#pragma endregion
  
