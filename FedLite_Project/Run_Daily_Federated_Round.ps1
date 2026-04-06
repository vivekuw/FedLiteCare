$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "FedLiteCare daily round starting..."
Write-Host "Mode: single-process nightly automation"

python .\Aggregator_Server\server\server_main.py --mode single-process

Write-Host "FedLiteCare daily round finished."
