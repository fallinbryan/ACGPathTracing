#include <optix.h>


#include "pathTracer.h"
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>


extern "C" {
  __constant__ PathTraceParams params;
}

struct Onb
{
  __forceinline__ __device__ Onb(const float3& normal)
  {
    m_normal = normal;

    if (fabs(m_normal.x) > fabs(m_normal.z))
    {
      m_binormal.x = -m_normal.y;
      m_binormal.y = m_normal.x;
      m_binormal.z = 0;
    }
    else
    {
      m_binormal.x = 0;
      m_binormal.y = -m_normal.z;
      m_binormal.z = m_normal.y;
    }

    m_binormal = normalize(m_binormal);
    m_tangent = cross(m_binormal, m_normal);
  }

  __forceinline__ __device__ void inverse_transform(float3& p) const
  {
    p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
  }

  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};



static __forceinline__ __device__ RadiancePayloadRayData loadClosesthitRadiancePRD()
{
  RadiancePayloadRayData prd = {};

  prd.attenuation.x = __uint_as_float(optixGetPayload_0());
  prd.attenuation.y = __uint_as_float(optixGetPayload_1());
  prd.attenuation.z = __uint_as_float(optixGetPayload_2());
  prd.randomSeed = optixGetPayload_3();
  prd.depth = optixGetPayload_4();
  prd.doneReason = (DoneReason)(optixGetPayload_18());
  return prd;
}

static __forceinline__ __device__ RadiancePayloadRayData loadMissRadiancePRD()
{
  RadiancePayloadRayData prd = {};
  return prd;
}

static __forceinline__ __device__ void storeClosesthitRadiancePRD(RadiancePayloadRayData prd)
{
  optixSetPayload_0(__float_as_uint(prd.attenuation.x));
  optixSetPayload_1(__float_as_uint(prd.attenuation.y));
  optixSetPayload_2(__float_as_uint(prd.attenuation.z));

  optixSetPayload_3(prd.randomSeed);
  optixSetPayload_4(prd.depth);

  optixSetPayload_5(__float_as_uint(prd.emissionColor.x));
  optixSetPayload_6(__float_as_uint(prd.emissionColor.y));
  optixSetPayload_7(__float_as_uint(prd.emissionColor.z));

  optixSetPayload_8(__float_as_uint(prd.radiance.x));
  optixSetPayload_9(__float_as_uint(prd.radiance.y));
  optixSetPayload_10(__float_as_uint(prd.radiance.z));

  optixSetPayload_11(__float_as_uint(prd.origin.x));
  optixSetPayload_12(__float_as_uint(prd.origin.y));
  optixSetPayload_13(__float_as_uint(prd.origin.z));

  optixSetPayload_14(__float_as_uint(prd.direction.x));
  optixSetPayload_15(__float_as_uint(prd.direction.y));
  optixSetPayload_16(__float_as_uint(prd.direction.z));

  optixSetPayload_17(prd.done);
  optixSetPayload_18(prd.doneReason);

}

static __forceinline__ __device__ void storeMissRadiancePRD(RadiancePayloadRayData prd)
{
  optixSetPayload_5(__float_as_uint(prd.emissionColor.x));
  optixSetPayload_6(__float_as_uint(prd.emissionColor.y));
  optixSetPayload_7(__float_as_uint(prd.emissionColor.z));

  optixSetPayload_8(__float_as_uint(prd.radiance.x));
  optixSetPayload_9(__float_as_uint(prd.radiance.y));
  optixSetPayload_10(__float_as_uint(prd.radiance.z));

  optixSetPayload_17(prd.done);
  optixSetPayload_18(prd.doneReason);
}

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
  // Uniformly sample disk.
  const float r = sqrtf(u1);
  const float phi = 2.0f * M_PIf * u2;
  p.x = r * cosf(phi);
  p.y = r * sinf(phi);

  // Project up to hemisphere.
  p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

static __forceinline__ __device__ bool uniform_sphere_rejection_sample(const float u1, const float u2, const float u3, float3& p)
{
  const float q = sqrtf(u1 * u1 + u2 * u2 + u3 * u3);
  if (q > 1.0f)
    return false;
  else {
    p.x = 2.0f * u1 * q;
    p.y = 2.0f * u2 * q;
    p.z = 2.0f * u3 * q;

    return true;
  
  }
}

static __forceinline__ __device__ const char* doneReasonToString(DoneReason reason)
{
  switch (reason)
  {
  case DoneReason::MISS:
    return "MISS";
  case DoneReason::MAX_DEPTH:
    return "MAX_DEPTH";
  case DoneReason::RUSSIAN_ROULETTE:
    return "RUSSIAN_ROULETTE";
  case DoneReason::NOT_DONE:
    return "NOT_DONE";
  default:
    return "UNKNOWN";
  }
}

static __forceinline__ __device__ bool pixelIsNull(const float3& pixel)
{
  return pixel.x == 0.0f && pixel.y == 0.0f && pixel.z == 0.0f;
}

static __forceinline__ __device__ float safeDivide(float a, float b)
{
  return b == 0.0f ? 0.0f : a / b;
}

static __forceinline__ __device__ float3 safeDivide(float3 a, float b) {
  return make_float3(safeDivide(a.x, b), safeDivide(a.y, b), safeDivide(a.z, b));

}

/**
 * @brief Traces a ray through the scene and updates the payload with the radiance information.
 *
 * This function uses the OptiX API to trace a ray through the scene and gather radiance information.
 * optixTraverse is used to trace the ray and optixInvoke is used to invoke the relavant shader.
 * The relvant shader will update the payload with the radiance and attenuation information.
 *
 * @param handle The handle to the traversable object (scene) to trace the ray through.
 * @param ray_origin The origin of the ray.
 * @param ray_direction The direction of the ray.
 * @param tmin The minimum t value for intersections.
 * @param tmax The maximum t value for intersections.
 * @param prd The payload to store the radiance information in.
 */
static __forceinline__ __device__ void traceRadiance(
  OptixTraversableHandle handle,
  float3                 ray_origin,
  float3                 ray_direction,
  float                  tmin,
  float                  tmax,
  RadiancePayloadRayData& prd
)
{
  unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18;

  u0 = __float_as_uint(prd.attenuation.x);
  u1 = __float_as_uint(prd.attenuation.y);
  u2 = __float_as_uint(prd.attenuation.z);
  u3 = prd.randomSeed;
  u4 = prd.depth;
  u18 = prd.doneReason;

  // Note:
  // This demonstrates the usage of the OptiX shader execution reordering 
  // (SER) API.  In the case of this computationally simple shading code, 
  // there is no real performance benefit.  However, with more complex shaders
  // the potential performance gains offered by reordering are significant.
  optixTraverse(
    RADIANCE_PAYLOAD_TYPE,
    handle,
    ray_origin,
    ray_direction,
    tmin,
    tmax,
    0.0f,                   // rayTime
    OptixVisibilityMask(1),
    OPTIX_RAY_FLAG_NONE,
    0,                      // SBT offset
    NUM_RAYTYPES,           // SBT stride
    0,                      // missSBTIndex
    u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18);

  optixReorder(
    // Application specific coherence hints could be passed in here
  );


  optixInvoke(
    RADIANCE_PAYLOAD_TYPE,
    u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18
  );

  prd.attenuation = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
  prd.randomSeed = u3;
  prd.depth = u4;

  prd.emissionColor = make_float3(__uint_as_float(u5), __uint_as_float(u6), __uint_as_float(u7));
  prd.radiance = make_float3(__uint_as_float(u8), __uint_as_float(u9), __uint_as_float(u10));
  prd.origin = make_float3(__uint_as_float(u11), __uint_as_float(u12), __uint_as_float(u13));
  prd.direction = make_float3(__uint_as_float(u14), __uint_as_float(u15), __uint_as_float(u16));
  prd.done = u17;
  prd.doneReason = (DoneReason)u18;
}


/**
 * @brief Traces a shadow ray through the scene and returns if the ray is occluded.
 *
 * This function uses the OptiX API to trace a shadow ray through the scene and returns if the ray is occluded.
 * optixTraverse is used to trace the ray and optixHitObjectIsHit is used to check if the ray hit an object.
 *
 * @param handle The handle to the traversable object (scene) to trace the ray through.
 * @param ray_origin The origin of the ray.
 * @param ray_direction The direction of the ray.
 * @param tmin The minimum t value for intersections.
 * @param tmax The maximum t value for intersections.
 * @return true if the ray is occluded, false otherwise.
 */
static __forceinline__ __device__ bool traceOcclusion(
  OptixTraversableHandle handle,
  float3                 ray_origin,
  float3                 ray_direction,
  float                  tmin,
  float                  tmax
)
{
  // We are only casting probe rays so no shader invocation is needed
  optixTraverse(
    handle,
    ray_origin,
    ray_direction,
    tmin,
    tmax, 0.0f,                // rayTime
    OptixVisibilityMask(1),
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    0,                         // SBT offset
    NUM_RAYTYPES,            // SBT stride
    0                          // missSBTIndex
  );
  return optixHitObjectIsHit();
}


/**
 * @brief Generates a ray for the current pixel and traces it through the scene.
 *
 * This function generates a ray for the current pixel and traces it through the scene.
 * The radiance information is accumulated and stored in the frame buffer.
 *
 * The function uses stratified sampling to generate rays for each pixel. Stratified sampling
 * is a method that divides the pixel into smaller sub-pixels and generates a sample within each sub-pixel.
 * This helps to reduce the variance and produce a more accurate image.
 *
 * The function also uses the Russian Roulette method for ray termination. This is a stochastic method
 * used to decide whether to continue or terminate a ray after each bounce. The decision is made based
 * on the intensity of the ray. If the ray's intensity is below a certain threshold, it has a certain
 * probability of being terminated. This helps to reduce the computational cost by not tracing rays
 * that contribute little to the final image.
 *
 * @param params The parameters for the path tracing, including the image width, height, camera parameters, 
 *               and the current frame index.
 */
extern "C" __global__ void __raygen__rg()
{
  
  const int    w = params.width;
  const int    h = params.height;
  const float3 eye = params.cameraEye;
  const float3 U = params.cameraU;
  const float3 V = params.cameraV;
  const float3 W = params.cameraW;
  const uint3  idx = optixGetLaunchIndex();
  const int    subframe_index = params.currentFrameIdx;
  const unsigned int maxDepth = params.maxDepth;
  

  unsigned int seed = tea<4>(idx.y * w + idx.x, subframe_index);

  float3 result = make_float3(0.0f);
  int i = params.samplesPerPixel;
  RadiancePayloadRayData prd;

  do
  {
    // The center of each pixel is at fraction (0.5,0.5)
    const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

    const float2 d = 2.0f * make_float2(
      (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(w),
      (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(h)
    ) - 1.0f;
    
    float3 ray_direction = normalize(d.x * U + d.y * V + W);
    float3 ray_origin = eye;


    // Initialize the payload for the first time
    prd.attenuation = make_float3(1.f); // The attenuation is initialized to 1
    prd.randomSeed = seed;
    prd.depth = 0;
    prd.doneReason = DoneReason::NOT_DONE;

    for (;; )
    {

      traceRadiance(
        params.handle,
        ray_origin,
        ray_direction,
        0.01f,  // tmin       
        1e16f,  // tmax
        prd
      );

      // at this point the PRD should be updated with the result of the trace
      result += prd.emissionColor;
      result += prd.radiance * prd.attenuation;

      const float p = dot(prd.attenuation, make_float3(0.30f, 0.59f, 0.11f)); // weight by perceived brightness to the human eye

      bool russianRoulette = rnd(prd.randomSeed) > p;

      const bool done = prd.done || russianRoulette || prd.depth >= maxDepth;
      if (done) {
        if(russianRoulette) prd.doneReason = DoneReason::RUSSIAN_ROULETTE;
        if(prd.depth >= maxDepth) prd.doneReason = DoneReason::MAX_DEPTH;
        break;
      }
      prd.attenuation = safeDivide(prd.attenuation, p);

      ray_origin = prd.origin;
      ray_direction = prd.direction;

      ++prd.depth;
    }
  } while (--i);

  const uint3    launch_index = optixGetLaunchIndex();
  const unsigned int image_index = launch_index.y * params.width + launch_index.x;
  float3         accum_color = result / static_cast<float>(params.samplesPerPixel);

  //if (
  //  pixelIsNull(accum_color) && 
  //  subframe_index > 500 && 
  //  prd.doneReason != DoneReason::MISS && 
  //  prd.doneReason != DoneReason::RUSSIAN_ROULETTE && 
  //  prd.doneReason != DoneReason::MAX_DEPTH
  //  )
  //{
  //  //char buffer[256];
  //  //const char* doneReason = doneReasonToString(prd.doneReason);
  //  //strncpy(buffer, doneReason, strlen(doneReason));
  //  printf("Current Depth: %d of %d\n", prd.depth, maxDepth);
  //  printf("Done Reason: %s\n", doneReasonToString(prd.doneReason));
  //  printf("result: %f, %f, %f\n", result.x, result.y, result.z);

  //}

  if (subframe_index > 0)
  {
    const float                 a = 1.0f / static_cast<float>(subframe_index + 1);
    const float3 accum_color_prev = make_float3(params.accumulationBuffer[image_index]);

    //the current frams is linear interpolated with the previous frames
    accum_color = lerp(accum_color_prev, accum_color, a);
  }
  params.accumulationBuffer[image_index] = make_float4(accum_color, 1.0f);
  
  //make_color is helper that clamps and gamma corrects the color into sRGB color space
  params.frameBuffer[image_index] = make_color(accum_color);

}


extern "C" __global__ void __miss__ms()
{
  
  optixSetPayloadTypes(RADIANCE_PAYLOAD_TYPE);

  MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
  RadiancePayloadRayData prd = loadMissRadiancePRD();

  prd.radiance = make_float3(rt_data->backgroundColor);
  prd.emissionColor = make_float3(0.f);
  prd.doneReason = DoneReason::MISS;
  prd.done = true;

  storeMissRadiancePRD(prd);
}

extern "C" __global__ void __closesthit__diffuse__ch()
{

  optixSetPayloadTypes(RADIANCE_PAYLOAD_TYPE);

  HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

  const int    prim_idx = optixGetPrimitiveIndex();
  const float3 ray_dir = optixGetWorldRayDirection();

  const uint3  idx = rt_data->indices[prim_idx];
  const bool useDirectLighting = params.useDirectLighting;
  const bool useImportanceSampling = params.useImportanceSampling;


  const float3 v0 = make_float3(rt_data->vertices[idx.x]);
  const float3 v1 = make_float3(rt_data->vertices[idx.y]);
  const float3 v2 = make_float3(rt_data->vertices[idx.z]);


  const float3 N_0 = normalize(cross(v1 - v0, v2 - v0));

  const float3 N = faceforward(N_0, -ray_dir, N_0);
  const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

  RadiancePayloadRayData prd = loadClosesthitRadiancePRD();

  if (prd.depth == 0)
    prd.emissionColor = rt_data->emissionColor;
  else
    prd.emissionColor = make_float3(0.0f);

  unsigned int seed = prd.randomSeed;
  {
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);


    float3 w_in;
    bool isRejected = false;
    if (useImportanceSampling)
    {
      cosine_sample_hemisphere(z1, z2, w_in);
      Onb onb(N);
      onb.inverse_transform(w_in);
    }
    else
    {
      float u1;
      float u2;
      float u3;
      do {
        u1 = rnd(seed);
        u2 = rnd(seed);
        u3 = rnd(seed);
      } while (!uniform_sphere_rejection_sample(u1, u2, u3, w_in));
      w_in += N;
      normalize(w_in);
    }

    prd.direction = w_in;
    prd.origin = P;

    prd.attenuation *= rt_data->diffuseColor;
    

  }

  const float z1 = rnd(seed);
  const float z2 = rnd(seed);
  prd.randomSeed = seed;

  AreaLight light = params.areaLight;
  float weight = 0.01f;
  if (useDirectLighting) 
  {
    weight = 0.0f;
    const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - P);
    const float3 L = normalize(light_pos - P);
    const float  nDl = dot(N, L);
    const float  LnDl = -dot(light.normal, L);

    if (nDl > 0.0f && LnDl > 0.0f)
    {
        const bool occluded =
          traceOcclusion(
            params.handle,
            P,
            L,
            0.01f,           // tmin
            Ldist - 0.01f);  // tmax

        if (!occluded)
        {
          const float A = length(cross(light.v1, light.v2));
          weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
          prd.radiance = light.emission * weight;
        }
      }                               
  }
  else {
    
    if (length(rt_data->emissionColor) > 0.0f) {
      prd.radiance = rt_data->emissionColor;
      prd.done = true;
      prd.doneReason = DoneReason::LIGHT_HIT;
    }
    else {
        prd.radiance = make_float3(0.0f);
        prd.done = false;
      }
    
  }


  storeClosesthitRadiancePRD(prd);
}