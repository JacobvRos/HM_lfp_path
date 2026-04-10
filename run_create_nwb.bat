@echo off
cd /d "%~dp0"
echo Current directory is: %cd%

call "C:\Users\Jacob\anaconda3\condabin\conda.bat" activate base

python create_nwb.py --usecd --sess_i 1 --sess_f 2 --ip nwb_data --op nwb_data

pause