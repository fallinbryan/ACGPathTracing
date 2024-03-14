//#include <util/tiny_obj_loader.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "TinyObjWrapper.h"
#include <iostream>

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>
#include <numeric>



TinyObjWrapper::TinyObjWrapper(const std::string& filename)
{
  loadFile(filename);
}


bool TinyObjWrapper::loadFile(const std::string& filename)
{
  std::string err;
  std::string warn;

  bool ret = reader.ParseFromFile(filename, reader_config);

  if (!ret) { 
    if(!reader.Error().empty()) std::cerr << "TinyObjReader: " << reader.Error();
   }
  if (!reader.Warning().empty()) {
    std::cout << "TinyObjReader: " << reader.Warning();
  }
  dataLoaded = ret;

  if (ret)
  {
    _updateVertices();
    _updateMaterials();
    _updateMaterialIndices();
    //reorderData();
  }

  return ret;
}

std::vector<Vertex> TinyObjWrapper::getVertices() const
{
  return _vertices;
}

std::vector<tinyobj::material_t> TinyObjWrapper::getMaterials() const
{

  return _materials;
}

std::vector<uint32_t> TinyObjWrapper::getMaterialIndices() const
{

  return _materialIndices;
}

size_t TinyObjWrapper::getNumMaterials() const
{
  return _materials.size();
}





void TinyObjWrapper::_updateMaterials()
{
  auto& materials = reader.GetMaterials();
  _materials = materials;
}

void TinyObjWrapper::_updateMaterialIndices()
{
  auto& shapes = reader.GetShapes();
  std::vector<uint32_t> materialIndices;
  if (dataLoaded)
  {
    for (auto& shape : shapes)
    {
      for (auto& index : shape.mesh.material_ids)
      {
        materialIndices.push_back(index);
      }
    }
  }
  _materialIndices = materialIndices;
}

void TinyObjWrapper::_updateVertices()
{
  auto& attrib = reader.GetAttrib();
  auto& shapes = reader.GetShapes();
  

  
  if (dataLoaded)
  {
    for (size_t s = 0; s < shapes.size(); s++) {
      size_t index_offset = 0;
      for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
        int fv = shapes[s].mesh.num_face_vertices[f];
        for (size_t v = 0; v < fv; v++) {
          tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
          Vertex vertex;
          vertex.x = attrib.vertices[3*idx.vertex_index+0];
          vertex.y = attrib.vertices[3*idx.vertex_index+1];
          vertex.z = attrib.vertices[3*idx.vertex_index+2];
          vertex.padding = 0.0f;
          _vertices.push_back(vertex);
        }
        index_offset += fv;
      }
    }
  }
  
}



uint64_t TinyObjWrapper::floatToFixed(float value, float minValue, float maxValue, int bits) {
  float scale = (std::pow(2, bits) - 1) / (maxValue - minValue);
  return static_cast<uint64_t>((value - minValue) * scale);
}

uint64_t TinyObjWrapper::mortonCode3D(uint64_t x, uint64_t y, uint64_t z) {
  uint64_t result = 0;
  for (uint64_t i = 0; i < (sizeof(uint64_t) * 8); ++i) {
    result |= (x & (1ULL << i)) << (2 * i) | (y & (1ULL << i)) << (2 * i + 1) | (z & (1ULL << i)) << (2 * i + 2);
  }
  return result;
}

std::vector<uint64_t> TinyObjWrapper::computeMortonCodes() {
  std::vector<uint64_t> mortonCodes(_vertices.size());
  // Determine bounding box to normalize vertex positions
  float minX = std::numeric_limits<float>::max(), minY = minX, minZ = minX;
  float maxX = std::numeric_limits<float>::lowest(), maxY = maxX, maxZ = maxX;
  for (const auto& vertex : _vertices) {
    minX = std::min(minX, vertex.x);
    minY = std::min(minY, vertex.y);
    minZ = std::min(minZ, vertex.z);
    maxX = std::max(maxX, vertex.x);
    maxY = std::max(maxY, vertex.y);
    maxZ = std::max(maxZ, vertex.z);
  }
  // Compute Morton codes
  for (size_t i = 0; i < _vertices.size(); ++i) {
    uint64_t x = floatToFixed(_vertices[i].x, minX, maxX, 21); // Using 21 bits for each dimension
    uint64_t y = floatToFixed(_vertices[i].y, minY, maxY, 21);
    uint64_t z = floatToFixed(_vertices[i].z, minZ, maxZ, 21);
    mortonCodes[i] = mortonCode3D(x, y, z);
  }
  return mortonCodes;
}



std::vector<size_t> TinyObjWrapper::computeSortedIndices() {
  std::vector<uint64_t> mortonCodes = computeMortonCodes();
  std::vector<size_t> indices(_vertices.size());
  std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., n-1
  std::sort(indices.begin(), indices.end(), [&mortonCodes](size_t i1, size_t i2) {
    return mortonCodes[i1] < mortonCodes[i2];
    });
  return indices;
}


void TinyObjWrapper::reorderData() {
  std::vector<size_t> sortedIndices = computeSortedIndices();

  // Reorder vertices
  std::vector<Vertex> sortedVertices(_vertices.size());
  for (size_t i = 0; i < _vertices.size(); ++i) {
    sortedVertices[i] = _vertices[sortedIndices[i]];
  }
  _vertices.swap(sortedVertices);

  // Reorder material indices if they are per-vertex
  if (!_materialIndices.empty()) {
    std::vector<uint32_t> sortedMaterialIndices(_materialIndices.size());
    for (size_t i = 0; i < _materialIndices.size(); ++i) {
      sortedMaterialIndices[i] = _materialIndices[sortedIndices[i]];
    }
    _materialIndices.swap(sortedMaterialIndices);
  }
}