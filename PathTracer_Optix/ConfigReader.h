#pragma once 

#include <string>
#include <map>


class ConfigReader
{
  std::map<std::string, std::string> _configMap;

public:
  ConfigReader(const std::string& configFileName);
  std::string getString(const std::string& key);
  int getInt(const std::string& key);
  
};