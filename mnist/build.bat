@echo off
echo Building Odin - Scratch
setlocal
cd %~dp0
odin build . -o:minimal -debug -ignore-unknown-attributes
echo Build Done at %time%