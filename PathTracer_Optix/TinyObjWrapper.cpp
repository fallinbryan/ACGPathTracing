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

  }

  return ret;
}

std::vector<float> TinyObjWrapper::getVerticesFloat() const
{
  return _vertices;
}

std::vector<Material> TinyObjWrapper::getMaterials() const
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


std::vector<uint32_t> TinyObjWrapper::getIndexBuffer() const
{
  return _indexBuffer;
}


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
    {
      mat.bsdfType = BSDFType::BSDF_DIFFUSE;
    }
  
    _materials.push_back(mat);
  } 
  
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

