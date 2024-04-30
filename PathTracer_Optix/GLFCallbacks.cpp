//#include <glad/glad.h>  // Needs to be included before gl_interop

#include "common.h"
#include "AppController.h"
#include "GLFCallbacks.h"
#include "tinylogger.h"



 void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);
  OptixManager* manager = AppController::getInstance().getOptixManager();

  if (action == GLFW_PRESS)
  {
    manager->ui.mouse_button = button;
    manager->ui.trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
  }
  else
  {
    manager->ui.mouse_button = -1;
  }
}

 void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
   OptixManager* manager = AppController::getInstance().getOptixManager();

  if (manager->ui.mouse_button == GLFW_MOUSE_BUTTON_LEFT)
  {
    manager->ui.trackball.setViewMode(sutil::Trackball::LookAtFixed);
    manager->ui.trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), manager->ui.width, manager->ui.height);
    manager->ui.camera_changed = true;
    
  }
  else if (manager->ui.mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
  {
    manager->ui.trackball.setViewMode(sutil::Trackball::EyeFixed);
    manager->ui.trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), manager->ui.width, manager->ui.height);
    manager->ui.camera_changed = true;
  }
}

 void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
   OptixManager* manager = AppController::getInstance().getOptixManager();
  if (manager->ui.trackball.wheelEvent((int)yscroll))
    manager->ui.camera_changed = true;
}

 void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{

  float stepSize = .105f;

  OptixManager* manager = AppController::getInstance().getOptixManager();

  if (action == GLFW_PRESS)
  {
    manager->ui.refreshAccumulationBuffer = true;

    switch (key)
    {
      case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, true);
        break;
      case GLFW_KEY_0:
        manager->toggle_direct_lighting();
        tlog::debug() << "Using Direct Lighting: " << (manager->use_direct_lighting() ? "yes" : "no");
        break;
      case GLFW_KEY_1:
        manager->toggle_importance_sampling();
        manager->ui.refreshAccumulationBuffer = true;
        tlog::debug() << "Using Importance Sampling: " << (manager->use_importance_sampling() ? "yes" : "no");
        break;
      case GLFW_KEY_UP:
        manager->increase_recursion_depth();
        tlog::debug() << "Ray Bounces: " << manager->max_depth();
        break;
      case GLFW_KEY_DOWN:
        manager->decrease_recurstion_depth();
        tlog::debug() << "Ray Bounces: " << manager->max_depth();
        break;
      case GLFW_KEY_R:
        break;

      default:
        manager->ui.refreshAccumulationBuffer = false;
        break;

    }
  }
}

