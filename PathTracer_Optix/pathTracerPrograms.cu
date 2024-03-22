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

//static __forceinline__ __device__ flaot fresnel


static __forceinline__ __device__ RadiancePayloadRayData loadClosesthitRadiancePRD()
{
  RadiancePayloadRayData prd = {};

  prd.attenuation.x = __uint_as_float(optixGetPayload_0());
  prd.attenuation.y = __uint_as_float(optixGetPayload_1());
  prd.attenuation.z = __uint_as_float(optixGetPayload_2());
  prd.randomSeed = optixGetPayload_3();
  prd.depth = optixGetPayload_4();
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


// Sample GGX distribution for importance sampling
static __forceinline__ __device__ float3 sampleGGX(float u1, float u2, float roughness, const float3& N)
{
  // Convert (u1, u2) uniform random variables into GGX distribution
  clamp(roughness, 0.001f, 1.0f); // Avoid division by zero (roughness = 0.0f is not allowed
  float phi = 2.0f * M_PIf * u1;
  float cosTheta = sqrtf((1.0f - u2) / (1.0f + (roughness * roughness - 1.0f) * u2));
  float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

  // Create sample vector in tangent space
  float3 H;
  H.x = sinTheta * cosf(phi);
  H.y = sinTheta * sinf(phi);
  H.z = cosTheta;

  // Transform H to world space
  float3 up = abs(N.z) < 0.999 ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
  float3 tangent = normalize(cross(up, N));
  float3 bitangent = cross(N, tangent);
  float3 sampleDir = H.x * tangent + H.y * bitangent + H.z * N;

  return normalize(sampleDir);
}

// Fresnel-Schlick approximation for conductors derived from the Pbr Book
static __forceinline__ __device__ float3 fresnelSchlickConductor(float cosTheta, float3 eta, float3 k)
{
  float3 eta2 = eta * eta;
  float3 k2 = k * k;

  float3 t1 = eta2 - k2 - make_float3(cosTheta * cosTheta);
  float3 a2plusb2 = make_float3(sqrtf(t1.x * t1.x + 4 * eta2.x * k2.x),
    sqrtf(t1.y * t1.y + 4 * eta2.y * k2.y),
    sqrtf(t1.z * t1.z + 4 * eta2.z * k2.z));

  float3 t2 = a2plusb2 + make_float3(cosTheta * cosTheta);

  float3 Rs = (t2 - 2 * eta * cosTheta + make_float3(cosTheta * cosTheta)) / (t2 + 2 * eta * cosTheta + make_float3(cosTheta * cosTheta));
  float3 Rp = Rs * (t2 - 2 * eta * cosTheta + make_float3(1)) / (t2 + 2 * eta * cosTheta + make_float3(1));

  return (Rs + Rp) * 0.5f;
}

// Fresnel for dialectrics derived from the Pbr Book
static __forceinline__ __device__ float FrDielectric(float cosThetaI, float etaI, float etaT) {
  cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
  // Flip the interface orientation if the incident ray is inside the material
  bool entering = cosThetaI > 0.0f;
  if (!entering) {
    // Swap etaI and etaT for rays inside the material
    float temp = etaI;
    etaI = etaT;
    etaT = temp;
    cosThetaI = fabs(cosThetaI);
  }

  float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
  float sinThetaT = etaI / etaT * sinThetaI;

  // Total internal reflection
  if (sinThetaT >= 1.0f) {
    return 1.0f; // When sinThetaT is greater or equal to 1, it indicates total internal reflection.
  }

  float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));

  float rParl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
  float rPerp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
  return (rParl * rParl + rPerp * rPerp) / 2.0f;
}




// GGX/Trowbridge-Reitz Normal Distribution Function
__forceinline__ __device__ float ggxNDF(float cosTheta, float roughness)
{
  clamp(roughness, 0.001f, 1.0f); // Avoid division by zero (roughness = 0.0f is not allowed
  float alpha = roughness * roughness;
  float denom = cosTheta * cosTheta * (alpha * alpha - 1.0f) + 1.0f;
  return (alpha * alpha) / (M_PIf * denom * denom);
}

// Schlick-GGX Geometric Shadowing
__forceinline__ __device__ float geometricSchlickGGX(float NdotV, float roughness)
{
  clamp(roughness, 0.001f, 1.0f); // Avoid division by zero (roughness = 0.0f is not allowed
  float r = (roughness + 1.0f);
  float k = (r * r) / 8.0f; // Beckmann approximation

  float denom = NdotV * (1.0f - k) + k;
  return NdotV / denom;
}

// Combined geometric shadowing for light and view directions
__forceinline__ __device__ float geometricSmith(float NdotV, float NdotL, float roughness)
{
  clamp(roughness, 0.001f, 1.0f); // Avoid division by zero (roughness = 0.0f is not allowed
  float ggxV = geometricSchlickGGX(NdotV, roughness);
  float ggxL = geometricSchlickGGX(NdotL, roughness);
  return ggxV * ggxL;
}

static __forceinline__ __device__ void traceRadiance(
  OptixTraversableHandle handle,
  float3                 ray_origin,
  float3                 ray_direction,
  float                  tmin,
  float                  tmax,
  RadiancePayloadRayData& prd
)
{
  unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17;

  u0 = __float_as_uint(prd.attenuation.x);
  u1 = __float_as_uint(prd.attenuation.y);
  u2 = __float_as_uint(prd.attenuation.z);
  u3 = prd.randomSeed;
  u4 = prd.depth;

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
    0.0f,                     // rayTime
    OptixVisibilityMask(1),
    OPTIX_RAY_FLAG_NONE,
    0,                        // SBT offset
    NUM_RAYTYPES,           // SBT stride
    0,                        // missSBTIndex
    u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17);
  optixReorder(
    // Application specific coherence hints could be passed in here
  );

  optixInvoke(RADIANCE_PAYLOAD_TYPE,
    u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17);

  prd.attenuation = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
  prd.randomSeed = u3;
  prd.depth = u4;

  prd.emissionColor = make_float3(__uint_as_float(u5), __uint_as_float(u6), __uint_as_float(u7));
  prd.radiance = make_float3(__uint_as_float(u8), __uint_as_float(u9), __uint_as_float(u10));
  prd.origin = make_float3(__uint_as_float(u11), __uint_as_float(u12), __uint_as_float(u13));
  prd.direction = make_float3(__uint_as_float(u14), __uint_as_float(u15), __uint_as_float(u16));
  prd.done = u17;
}

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
  if (optixHitObjectIsHit()) {
    // get the object that was hit 
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    if (rt_data->bsdfType == BSDFType::BSDF_REFRACTION) {
      return false;
    }
    else {
      return true;
    }
  }

  return false;
}


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


  unsigned int seed = tea<4>(idx.y * w + idx.x, subframe_index);

  float3 result = make_float3(0.0f);
  int i = params.samplesPerPixel;

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

    RadiancePayloadRayData prd;
    prd.attenuation = make_float3(1.f);
    prd.randomSeed = seed;
    prd.depth = 0;

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

      result += prd.emissionColor;
      result += prd.radiance * prd.attenuation;

      const float p = dot(prd.attenuation, make_float3(0.30f, 0.59f, 0.11f));
      const bool done = prd.done || rnd(prd.randomSeed) > p;
      if (done) {

        break;
      }
      prd.attenuation /= p;

      ray_origin = prd.origin;
      ray_direction = prd.direction;

      ++prd.depth;
    }
  } while (--i);
  //printf("result: %f, %f, %f\n", result.x, result.y, result.z);
  const uint3    launch_index = optixGetLaunchIndex();
  const unsigned int image_index = launch_index.y * params.width + launch_index.x;
  float3         accum_color = result / static_cast<float>(params.samplesPerPixel);

  if (subframe_index > 0)
  {
    const float                 a = 1.0f / static_cast<float>(subframe_index + 1);
    const float3 accum_color_prev = make_float3(params.accumulationBuffer[image_index]);
    accum_color = lerp(accum_color_prev, accum_color, a);
  }
  params.accumulationBuffer[image_index] = make_float4(accum_color, 1.0f);
  params.frameBuffer[image_index] = make_color(accum_color);

}


extern "C" __global__ void __miss__ms()
{

  optixSetPayloadTypes(RADIANCE_PAYLOAD_TYPE);

  MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
  RadiancePayloadRayData prd = loadMissRadiancePRD();

  prd.radiance = make_float3(rt_data->backgroundColor);
  prd.emissionColor = make_float3(0.f);
  prd.done = true;

  storeMissRadiancePRD(prd);
}

extern "C" __global__ void __closesthit__ch()
{

  optixSetPayloadTypes(RADIANCE_PAYLOAD_TYPE);

  HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

  const int       prim_idx = optixGetPrimitiveIndex();
  const float3    ray_dir = optixGetWorldRayDirection();

  const uint3     idx = rt_data->indices[prim_idx];
  const bool      useDirectLighting = params.useDirectLighting;
  const float     metallic = rt_data->metallic;
  const float     roughness =  rt_data->roughness;
  const float     IOR = rt_data->IOR;
  const BSDFType  bsdfType = rt_data->bsdfType;


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

  switch (bsdfType)
  {
  case BSDFType::BSDF_DIFFUSE:
  {
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    float3 w_in;
    cosine_sample_hemisphere(z1, z2, w_in);
    Onb onb(N);
    onb.inverse_transform(w_in);

    prd.direction = w_in;
    prd.origin = P;
    prd.attenuation *= rt_data->diffuseColor;
    break;
  }
  case BSDFType::BSDF_METALLIC:
  {

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    float3 microfacetNormal = sampleGGX(z1, z2, roughness, N);
    float3 R = reflect(ray_dir, microfacetNormal); /// Refelction should have a random portion as well

    prd.direction = R;
    prd.origin = P + R * 1e-4f;


    float3 eta = make_float3(1.45, 0.7, 1.55); // Slightly more refraction in the blue channel
    float3 k = make_float3(3.0, 2.2, 3.5); // Higher absorption in the red and blue channels
    float cosTheta = fmaxf(dot(microfacetNormal, -ray_dir), 0.0f);
    float3 F = fresnelSchlickConductor(cosTheta, eta, k);
    float3 F0 = rt_data->diffuseColor;
    float3 color = F * F0;

    prd.attenuation *= color;
    break;

  }
  case BSDFType::BSDF_REFRACTION:
  {

    float3 incidentRayDir = normalize(ray_dir);

    float cos_theta = dot(normalize(-ray_dir), N_0);
    float F = FrDielectric(cos_theta, 1.0f, IOR);


    if (rnd(seed) < F) {
      prd.direction = reflect(incidentRayDir, N_0);

    }
    else {

      float3 refractedDir; // Initialized by the refract function
      bool didRefract = refract(refractedDir, incidentRayDir, N_0, IOR);
      if (didRefract) {
        prd.direction = refractedDir;
      }
      else {
        prd.direction = reflect(incidentRayDir, N_0);
      }
    }
    prd.origin = P + prd.direction * 1e-3f;
    prd.attenuation *= rt_data->diffuseColor;
    break;

  }
  }

  const float z1 = rnd(seed);
  const float z2 = rnd(seed);
  prd.randomSeed = seed;

  float weight = 0.01f;
  AreaLight light = params.areaLight;
  if (useDirectLighting)
  {
    weight = 0.0f;
    //perturb the light position
    const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - P);
    const float3 L = normalize(light_pos - P);
    const float  nDl = dot(N, L);
    const float  LnDl = -dot(light.normal, L);

        if (nDl > 0.0f && LnDl > 0.0f)
        {
          const bool occluded = traceOcclusion(params.handle,P,L,0.01f, Ldist - 0.01f);  

          if (!occluded)
          {
            const float A = length(cross(light.v1, light.v2));
           

            switch(bsdfType)
            {
              case::BSDF_METALLIC:
              {
                //const float3 reflected = reflect(-L, N_0);

                // Compute the halfway vector between the light direction and the view direction
                const float3 H = normalize(L - ray_dir);
                const float NdotH = fmaxf(dot(N_0, H), 0.0f);
                const float NdotL = fmaxf(dot(N_0, L), 0.0f);

                // Calculate the distribution of microfacets based on roughness
                const float D = ggxNDF(NdotH, roughness);
                //printf("D: %f\n", D);

                // Calculate the Fresnel term using Schlick's approximation
                const float3 F = fresnelSchlickConductor(fmaxf(dot(H, ray_dir), 0.0f), make_float3(1.45, 0.7, 1.55), make_float3(3.0, 2.2, 3.5));

                //printf("F: %f, %f, %f\n", F.x, F.y, F.z);

                // Combine the normal distribution function and Fresnel term
                float specularContribution = D * F.x;

                //printf("specularContribution: %f\n", specularContribution);

                const float G = geometricSmith(fmaxf(dot(N_0, ray_dir), 0.0f), fmaxf(dot(N_0, L), 0.0f), roughness);
              
                ///printf("G: %f\n", G);

                // Modulate by the specular term and the light's contribution
                weight = NdotL * specularContribution * G;
                const float3 radiance = light.emission * weight / (Ldist * Ldist);

                //printf("radiance: %f, %f, %f\n", radiance.x, radiance.y, radiance.z);
                prd.radiance += radiance;

                break;
              }
              default:
              {
                weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
                prd.radiance = light.emission * weight;
                break;
              }
            }

          }
        
      
    }
  }
  else {
    prd.radiance = light.emission * weight;
  }
  prd.done = false;

  storeClosesthitRadiancePRD(prd);
}


extern "C" __global__ void __anyhit__ah()
{
  // For a pass-through any-hit shader
  // Just return, letting the ray continue to the closest-hit or miss shader.
}
