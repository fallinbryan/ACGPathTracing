//#include <util/tiny_obj_loader.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "TinyObjWrapper.h"
#include <vector_types.h>
#include <iostream>



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
  return ret;
}

std::vector<Vertex> TinyObjWrapper::getVertices() const
{
  auto& attrib = reader.GetAttrib();
  auto& shapes = reader.GetShapes();
  

  std::vector<Vertex> vertices;
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
          vertices.push_back(vertex);
        }
        index_offset += fv;
      }
    }
  }
  return vertices;
}

std::vector<float3> TinyObjWrapper::getMaterials() const
{
  auto& materials = reader.GetMaterials();
  std::vector<float3> materialsVec;
  if (dataLoaded)
  {
    for (auto& material : materials)
    {
      float3 mat;
      mat.x = material.diffuse[0];
      mat.y = material.diffuse[1];
      mat.z = material.diffuse[2];
      materialsVec.push_back(mat);
    }
  }
  return materialsVec;
}

std::vector<uint32_t> TinyObjWrapper::getMaterialIndices() const
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
  return materialIndices;
}

size_t TinyObjWrapper::getNumMaterials() const
{
  auto& materials = reader.GetMaterials();
  return materials.size();
}