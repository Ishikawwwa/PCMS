param([switch]$Install, [switch]$Embeddings, [string]$Python = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
if ($Install) {
    $syncArgs = @('sync', '--project', 'mnemos', '--extra', 'dev')
    if ($Embeddings) { $syncArgs += @('--extra', 'local-embeddings') }
    & uv @syncArgs
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
    Push-Location (Join-Path $PSScriptRoot 'frontend')
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
}
if (-not $Python) {
    $runtimeConfig = Join-Path $PSScriptRoot 'state\runtime.json'
    if (Test-Path $runtimeConfig) { $Python = (Get-Content -Raw $runtimeConfig | ConvertFrom-Json).python }
}
if (-not $Python) {
    $appPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    $repoPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (Test-Path $appPython) { $Python = $appPython }
    elseif (Test-Path $repoPython) { $Python = $repoPython }
    else { throw 'Run .\mnemos\start.ps1 -Install, or pass -Python with an environment containing the app dependencies.' }
}
if (-not (Test-Path (Join-Path $PSScriptRoot 'frontend\dist\index.html'))) {
    throw 'Build the interface first: npm --prefix mnemos/frontend run build'
}
& $Python -c 'import fastapi,uvicorn,numpy'
if ($LASTEXITCODE -ne 0) { throw 'Missing Python dependencies. Run .\mnemos\start.ps1 -Install' }
Write-Host 'Mnemos: http://127.0.0.1:8000  (Ctrl+C to stop)' -ForegroundColor Cyan
& $Python -m uvicorn mnemos.backend.app:app --host 127.0.0.1 --port 8000
