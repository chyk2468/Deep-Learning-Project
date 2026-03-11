@echo off

set FILE=.checkpoint

if not exist %FILE% (
    echo 0 > %FILE%
)

set /p num=<%FILE%
set /a num=num+1

echo %num% > %FILE%

git add .
git commit -m "checkpoint %num%"

echo.
echo Commit created: checkpoint %num%
pause
