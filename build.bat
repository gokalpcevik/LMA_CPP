@echo off
call cmake --build build --target ALL_BUILD --config %~1 -j 8 
