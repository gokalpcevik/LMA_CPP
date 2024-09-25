@echo off
mkdir build
call cmake --no-warn-unused-cli -DCMAKE_TOOLCHAIN_FILE:STRING=C:\vcpkg\scripts\buildsystems\vcpkg.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE "-Bbuild" -DVCPKG_TARGET_TRIPLET=x64-windows
