# ICU Drug Env — Setup and Test Script
# Run this in PowerShell to install dependencies and run all tests

Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host "Step 1: Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "=" * 50 -ForegroundColor Cyan

& "C:/Users/ABDUL AZEEZ/AppData/Local/Microsoft/WindowsApps/python3.13.exe" -m pip install --upgrade pip
& "C:/Users/ABDUL AZEEZ/AppData/Local/Microsoft/WindowsApps/python3.13.exe" -m pip install pydantic fastapi "uvicorn[standard]" openai websockets httpx python-multipart

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: pip install failed. Trying alternative approach..." -ForegroundColor Red
    & "C:/Users/ABDUL AZEEZ/AppData/Local/Microsoft/WindowsApps/python3.13.exe" -m pip install pydantic
    & "C:/Users/ABDUL AZEEZ/AppData/Local/Microsoft/WindowsApps/python3.13.exe" -m pip install fastapi
    & "C:/Users/ABDUL AZEEZ/AppData/Local/Microsoft/WindowsApps/python3.13.exe" -m pip install uvicorn
    & "C:/Users/ABDUL AZEEZ/AppData/Local/Microsoft/WindowsApps/python3.13.exe" -m pip install openai
    & "C:/Users/ABDUL AZEEZ/AppData/Local/Microsoft/WindowsApps/python3.13.exe" -m pip install websockets
    & "C:/Users/ABDUL AZEEZ/AppData/Local/Microsoft/WindowsApps/python3.13.exe" -m pip install httpx
    & "C:/Users/ABDUL AZEEZ/AppData/Local/Microsoft/WindowsApps/python3.13.exe" -m pip install python-multipart
}

Write-Host "`n" 
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host "Step 2: Running smoke tests..." -ForegroundColor Yellow
Write-Host "=" * 50 -ForegroundColor Cyan

& "C:/Users/ABDUL AZEEZ/AppData/Local/Microsoft/WindowsApps/python3.13.exe" "c:/Users/ABDUL AZEEZ/Downloads/icu-drug-env (1)/icu-drug-env/test_local.py"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nALL TESTS PASSED!" -ForegroundColor Green
} else {
    Write-Host "`nSome tests failed. Check output above." -ForegroundColor Red
}
