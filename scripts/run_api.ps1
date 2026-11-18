# Laptop 1: PathSnap API Host (PowerShell)
param(
    [int]$Port = 8000,
    [string]$HostAddress = "0.0.0.0"
)

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  PathSnap Backend Server (Laptop 1: Model Host)  " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Binding to: $HostAddress:$Port" -ForegroundColor Green
Write-Host "Contract: docs/API_CONTRACT.md" -ForegroundColor Gray
Write-Host ""

# Ensure firewall allows inbound traffic on the selected port
Write-Host "Note: To allow inbound network traffic on Laptop 1, run once as Administrator:" -ForegroundColor Yellow
Write-Host "  netsh advfirewall firewall add rule name=""PathSnap API"" dir=in action=allow protocol=TCP localport=$Port" -ForegroundColor Yellow
Write-Host ""

python -m uvicorn api.main:app --host $HostAddress --port $Port
