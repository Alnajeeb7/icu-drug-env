Write-Host "Syncing to GitHub and Hugging Face..." -ForegroundColor Cyan

# 1. Add all changes
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
