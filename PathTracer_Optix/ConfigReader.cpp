#include "ConfigReader.h"

#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include <map>
#include <string_view>

std::string trim(const std::string& str) {
  auto wsfront = std::find_if_not(str.begin(), str.end(), [](int c) { return std::isspace(c); });
  auto wsback = std::find_if_not(str.rbegin(), str.rend(), [](int c) { return std::isspace(c); }).base();
  return (wsback <= wsfront ? std::string() : std::string(wsfront, wsback));
}

namespace fs = std::filesystem;

ConfigReader::ConfigReader(const std::string& configFileName)
{

  fs::path exePath = fs::current_path();
  fs::path filePath = exePath / configFileName;

  std::ifstream configFile(filePath);
  
  if (!configFile.is_open())
  {
    std::cerr << "Error: could not open config file " << configFileName << std::endl;
    return;
  }

  std::string line;
  while (std::getline(configFile, line))
  {
    if (line.empty() || line[0] == '#')
    {
      continue;
    }
    size_t pos = line.find('=');
    if (pos == std::string::npos)
    {
      std::cerr << "Error: invalid line in config file: " << line << std::endl;
      continue;
    }

    std::string key = trim(line.substr(0, pos));
    std::string value = trim(line.substr(pos + 1));
    _configMap[key] = value;
  }
}

std::string ConfigReader::getString(const std::string& key)
{
  return _configMap[key];
}

int ConfigReader::getInt(const std::string& key)
{
  return std::stoi(_configMap[key]);
}
