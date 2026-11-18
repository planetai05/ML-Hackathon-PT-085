# Laptop 2: PathSnap Evaluator UI (PowerShell)
param(
    [string]$ApiUrl = "http://127.0.0.1:8000",
    [int]$Port = 8501
)

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  PathSnap Evaluator UI (Laptop 2: Client)       " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Target API URL: $ApiUrl" -ForegroundColor Green
Write-Host "UI Server Port: $Port" -ForegroundColor Green
Write-Host ""

$env:PATHSNAP_API_URL = $ApiUrl
python -m streamlit run ui/app.py --server.port $Port --server.address 0.0.0.0
