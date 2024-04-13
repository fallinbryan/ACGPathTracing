#pragma once

#include <optix.h>
#include "TinyObjWrapper.h"

constexpr unsigned int NUM_RAYTYPES = 1; // TODO: Add shadow ray type

constexpr OptixPayloadTypeID RADIANCE_PAYLOAD_TYPE = OPTIX_PAYLOAD_TYPE_ID_0;
constexpr OptixPayloadTypeID SHADOW_PAYLOAD_TYPE = OPTIX_PAYLOAD_TYPE_ID_1;

enum DoneReason {
  MISS,
  MAX_DEPTH,
  RUSSIAN_ROULETTE,
  LIGHT_HIT,
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
  
};

struct ShadowPayloadRayData {
  bool inShadow;
};

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

const unsigned int shadowPayloadRayDataSemantics[1] =
{
  // ShadowPayloadRayData::inShadow
  OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE 
};


struct AreaLight {
  float3 corner;
  float3 v1;
  float3 v2;
  float3 normal;
  float3 emission;
};

struct PathTraceParams {

  unsigned int currentFrameIdx;
  float4* accumulationBuffer;
  uchar4* frameBuffer;

  unsigned int width;
  unsigned int height;
  unsigned int samplesPerPixel;
  unsigned int maxDepth;

  float3 cameraEye;
  float3 cameraU;
  float3 cameraV;
  float3 cameraW;

  AreaLight areaLight;
  OptixTraversableHandle handle;

  bool useDirectLighting;
  bool useImportanceSampling;

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
  float3 emissionColor;
  float3 diffuseColor;
  float IOR;
  float roughness;
  float metallic;
  float4* vertices;
  uint3* indices;
};