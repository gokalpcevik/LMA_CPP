@echo off
mkdir compile_commands 
call cmake --no-warn-unused-cli -DCMAKE_TOOLCHAIN_FILE:STRING=C:\vcpkg\scripts\buildsystems\vcpkg.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -GNinja "-Bcompile_commands" -DVCPKG_TARGET_TRIPLET=x64-windows
call cmake --build compile_commands -j 16 --config Debug
xcopy "compile_commands\compile_commands.json" "compile_commands.json"
rmdir /s /q compile_commands
