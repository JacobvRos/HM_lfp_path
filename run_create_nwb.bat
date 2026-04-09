@echo off
cd /d "%~dp0"
echo Current directory is: %cd%

call "C:\Users\Jacob\anaconda3\condabin\conda.bat" activate base

python create_nwb.py --usecd --ip nwb_data --op nwb_data

pause