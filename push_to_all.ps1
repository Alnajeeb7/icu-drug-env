Write-Host "Syncing to GitHub and Hugging Face..." -ForegroundColor Cyan

# 0. Ensure uv is installed (REQUIRED for OpenEnv)
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv package manager..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path += ";$HOME\.cargo\bin"
}

# 1. Generate uv.lock
Write-Host "Generating uv.lock..." -ForegroundColor Yellow
uv lock

# 2. Add all changes
git add .

# 2. Commit
$commitMsg = "Finalize OpenEnv submission and unified deployment"
git commit -m $commitMsg

# 3. Push to GitHub (origin)
Write-Host "Pushing to GitHub (origin)..." -ForegroundColor Green
git push origin main

# 4. Push to Hugging Face (hf)
Write-Host "Pushing to Hugging Face (hf)..." -ForegroundColor Yellow
git push hf main

Write-Host "Done! Updates pushed to both remotes." -ForegroundColor Cyan
