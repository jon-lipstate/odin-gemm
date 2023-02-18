@echo off
echo Building Odin - GEMM
setlocal
cd %~dp0
odin build . -o:speed
echo Build Done at %time%