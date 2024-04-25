//#include <util/tiny_obj_loader.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "TinyObjWrapper.h"
#include <iostream>

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>
#include <numeric>
#include <optix_types.h>
#include <algorithm>

enum class VectorIDX
{
  X = 0,
  Y = 1,
  Z = 2
};

float getValFromFloat3(VectorIDX idx, const float3& v)
{
  switch (idx)
  {
  case VectorIDX::X:
    return v.x;
    break;
  case VectorIDX::Y:
    return v.y;
    break;
  case VectorIDX::Z:
    return v.z;
    break;
  default:
    throw std::runtime_error("Invalid VectorIDX");
    break;
  }
}

float getMin(VectorIDX idx, const std::vector<float3>& vertices)
{
  float min = std::numeric_limits<float>::max();

  for (auto& v : vertices)
  {
    float val = getValFromFloat3(idx, v);
    min = std::min(min, val);
  }
  return min;
}

float getMax(VectorIDX idx, const std::vector<float3>& vertices)
{
  float max = std::numeric_limits<float>::min();

  for (auto& v : vertices)
  {
    float val = getValFromFloat3(idx, v);
    max = std::max(max, val);
  }
  return max;
}

OptixAabb GetAABBFromVerts(const std::vector<float3>& vertices)
{
  
  OptixAabb aabb;
  aabb.maxX = getMax(VectorIDX::X, vertices);
  aabb.maxY = getMax(VectorIDX::Y, vertices);
  aabb.maxZ = getMax(VectorIDX::Z, vertices);
  aabb.minX = getMin(VectorIDX::X, vertices);
  aabb.minY = getMin(VectorIDX::Y, vertices);
  aabb.minZ = getMin(VectorIDX::Z, vertices);
  return aabb;
}



/**
 * @brief Constructor for the TinyObjWrapper class that takes a filename as an argument.
 *
 * This constructor calls the loadFile method to load the .obj file specified by the filename.
 * If the file is successfully loaded, the constructor will also update the vertices, materials, material indices, and index buffer.
 *
 * @param filename The name of the .obj file to load.
 */
TinyObjWrapper::TinyObjWrapper(const std::string& filename)
{
  loadFile(filename);
}


/**
 * @brief Loads an .obj file and updates the vertices, materials, material indices, and index buffer.
 *
 * This method uses the tinyobj library to parse the .obj file specified by the filename.
 * If the file is successfully parsed, the method will update the vertices, materials, material indices, and index buffer.
 * The method also sets the triangulate and vertex_color options of the reader_config to true and false, respectively.
 *
 * @param filename The name of the .obj file to load.
 * @return A boolean indicating whether the file was successfully loaded.
 */

bool TinyObjWrapper::loadFile(const std::string& filename)
{
  std::string err;
  std::string warn;

  reader_config.triangulate = true;
  reader_config.vertex_color = false;

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
    _updateIndexBuffer();
    _updateAabbs();

  }

  return ret;
}

/**
 * @brief Returns the vertices of the 3D model as a vector of floats.
 *
 * This method returns the _vertices member variable, which is updated by the loadFile method.
 * Each vertex is represented by four floats (x, y, z, w), where w is always 1.0.
 *
 * @return A vector of floats representing the vertices of the 3D model.
 */
std::vector<float> TinyObjWrapper::getVerticesFloat() const
{
  return _vertices;
}

/**
 * @brief Returns the materials of the 3D model as a vector of Material structs.
 *
 * This method returns the _materials member variable, which is updated by the loadFile method.
 *
 * @return A vector of Material structs representing the materials of the 3D model.
 */
std::vector<Material> TinyObjWrapper::getMaterials() const
{

  return _materials;
}

/**
 * @brief Returns the material indices of the 3D model as a vector of uint32_t.
 *
 * This method returns the _materialIndices member variable, which is updated by the loadFile method.
 *
 * @return A vector of uint32_t representing the material indices of the 3D model.
 */
std::vector<uint32_t> TinyObjWrapper::getMaterialIndices() const
{

  return _materialIndices;
}

/**
 * @brief Returns the number of materials of the 3D model.
 *
 * This method returns the size of the _materials member variable, which is updated by the loadFile method.
 *
 * @return The number of materials of the 3D model.
 */
size_t TinyObjWrapper::getNumMaterials() const
{
  return _materials.size();
}


/**
 * @brief Returns the index buffer of the 3D model as a vector of uint32_t.
 *
 * This method returns the _indexBuffer member variable, which is updated by the loadFile method.
 *
 * @return A vector of uint32_t representing the index buffer of the 3D model.
 */
std::vector<uint32_t> TinyObjWrapper::getIndexBuffer() const
{
  return _indexBuffer;
}

/**
 * @brief Updates the materials associated with the 3D model.
 *
 * This method updates the _materials member variable by iterating over the materials returned by the tinyobj library.
 * Each material is converted to a Material struct and added to the _materials vector.
 */
void TinyObjWrapper::_updateMaterials()
{
  auto& materials = reader.GetMaterials();
  Material mat;
  for (auto m : materials)
  {
    mat.diffuse = {m.diffuse[0], m.diffuse[1], m.diffuse[2]};
    mat.emission = {m.emission[0], m.emission[1], m.emission[2]};
    mat.roughness = m.roughness;
    mat.metallic = m.metallic;
    mat.ior = m.ior;

    // set the bsdf type based on the material name.  if the name contains Refractive , then it's a refraction material
    // if the name contains Metallic, then it's a metallic material
    // otherwise, it's a diffuse material
    if (m.name.find("Refractive") != std::string::npos)
    {
      mat.bsdfType = BSDFType::BSDF_REFRACTION;
    }
    else if (m.name.find("Metallic") != std::string::npos)
    {
      mat.bsdfType = BSDFType::BSDF_METALLIC;
    }
    else
    if (m.name.find("Volume") != std::string::npos)
    {
      mat.bsdfType = BSDFType::BSDF_VOLUME;
      mat.volumeDensity = m.roughness;
    }
    else
    {
      mat.bsdfType = BSDFType::BSDF_DIFFUSE;
    }
  
    _materials.push_back(mat);
  } 
  
}

/**
 * @brief Updates the material indices associated with the 3D model.
 *
 * This method updates the _materialIndices member variable by iterating over the shapes returned by the tinyobj library.
 * Each shape contains a mesh with material_ids, which are added to the _materialIndices vector.
 */
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

/**
 * @brief Updates the vertices associated with the 3D model.
 *
 * This method updates the _vertices member variable by iterating over the vertices returned by the tinyobj library.
 * Each vertex is represented by four floats (x, y, z, w), where w is always 1.0.
 */
void TinyObjWrapper::_updateVertices()
{
  auto& attrib = reader.GetAttrib();
  auto& shapes = reader.GetShapes();
  

  
  if (dataLoaded)
  {
    _vertices.reserve((attrib.vertices.size() / 3) * 4);
    for (size_t i = 0; i < attrib.vertices.size(); i += 3)
    {
      _vertices.push_back(attrib.vertices[i]);
      _vertices.push_back(attrib.vertices[i + 1]);
      _vertices.push_back(attrib.vertices[i + 2]);
      _vertices.push_back(1.0f);
    }
   
 
  }
  
}

/**
 * @brief Updates the index buffer associated with the 3D model.
 *
 * This method updates the _indexBuffer member variable by iterating over the shapes returned by the tinyobj library.
 * Each shape contains a mesh with indices, which are added to the _indexBuffer vector.
 */
void TinyObjWrapper::_updateIndexBuffer()
{
  auto& shapes = reader.GetShapes();
  std::vector<uint32_t> indexBuffer;
  if (dataLoaded)
  {
    for (auto& shape : shapes)
    {
      for (auto& index : shape.mesh.indices)
      {
        indexBuffer.push_back(index.vertex_index);
      }
    }
  }
  _indexBuffer = indexBuffer;
}


std::vector<OptixAabb> TinyObjWrapper::getAabbs() const
{
  return _aabbs;
}

void TinyObjWrapper::_updateAabbs() {
    auto& shapes = reader.GetShapes();
    std::vector<OptixAabb> aabbs;
    if (dataLoaded)
    {
      for (auto& shape : shapes)
      {
        std::vector<float3> vertices;
        for (auto& index : shape.mesh.indices)
        {
          float3 v = {reader.GetAttrib().vertices[index.vertex_index * 3], reader.GetAttrib().vertices[index.vertex_index * 3 + 1], reader.GetAttrib().vertices[index.vertex_index * 3 + 2]};
          vertices.push_back(v);
        }
        aabbs.push_back(GetAABBFromVerts(vertices));
      }
    }
  _aabbs = aabbs;
}
