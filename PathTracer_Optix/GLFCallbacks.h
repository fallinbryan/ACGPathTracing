#pragma once GLFCALLBACKS_H
#include "common.h"


 void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

 void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);

 void scrollCallback(GLFWwindow* window, double xscroll, double yscroll);

 void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/);