$ErrorActionPreference = "Stop"
param(
    [string]$TaskName = "FedLiteCare_Daily_Round",
    [string]$StartTime = "00:00"
)

Set-Location $PSScriptRoot

$scriptPath = Join-Path $PSScriptRoot "Run_Daily_Federated_Round.ps1"
$quotedScriptPath = '"' + $scriptPath + '"'
$taskCommand = "powershell.exe -ExecutionPolicy Bypass -File $quotedScriptPath"

Write-Host "Registering Windows Scheduled Task..."
Write-Host "Task Name: $TaskName"
Write-Host "Start Time: $StartTime"
Write-Host "Command: $taskCommand"

schtasks /Create /SC DAILY /ST $StartTime /TN $TaskName /TR $taskCommand /F

Write-Host "Scheduled task created successfully."
