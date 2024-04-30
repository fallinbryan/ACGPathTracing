#ifndef APP_CONTROLLER_H
#define APP_CONTROLLER_H

#include "OptixManager.h"
#include <memory>

class AppController {
public:
  static AppController& getInstance() {
    static AppController instance;
    return instance;
  }

  OptixManager* getOptixManager() const {
    if (optixManager)
      return optixManager.get();
    else {
      throw std::runtime_error("OptixManager not initialized");
    }
  }

  OptixManager* setOptixManager(const OptixSettings& settings)  {
    if (!optixManager)
      optixManager = std::make_unique<OptixManager>(settings);
    return optixManager.get();
  }

  AppController(AppController const&) = delete;
  AppController& operator=(AppController const&) = delete;

private:
  AppController() : optixManager(nullptr) {}

  std::unique_ptr<OptixManager> optixManager;
};

#endif // APP_CONTROLLER_H
