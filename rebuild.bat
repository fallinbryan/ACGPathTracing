@echo off
echo Cleaning up the build directory...
rmdir /S /Q build 2>nul
if %ERRORLEVEL% NEQ 0 echo Failed to remove build directory, it may not exist. Continuing...

echo Creating new build directory...
mkdir build
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create build directory.
    exit /b %ERRORLEVEL%
)

echo Changing to build directory...
cd build
if %ERRORLEVEL% NEQ 0 (
    echo Failed to change directory to build.
    exit /b %ERRORLEVEL%
)

echo Configuring the project with CMake...
cmake .. >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Configuration failed.
    exit /b %ERRORLEVEL%
)

echo Building the project...
cmake --build . --config Release -- /verbosity:normal > build_output.txt 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Build failed. See build_output.txt for details.
    exit /b %ERRORLEVEL%
)

echo Build completed successfully. Output is in build_output.txt
pause
