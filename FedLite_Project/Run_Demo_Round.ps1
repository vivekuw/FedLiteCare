$ErrorActionPreference = "Stop"
param(
    [int]$StartupDelaySeconds = 8
)

Set-Location $PSScriptRoot

function Start-DemoWindow {
    param(
        [string]$Title,
        [string]$Command
    )

    $escapedRoot = $PSScriptRoot.Replace("'", "''")
    $fullCommand = "& { `$Host.UI.RawUI.WindowTitle = '$Title'; Set-Location '$escapedRoot'; $Command }"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $fullCommand | Out-Null
}

Start-DemoWindow `
    -Title "FedLiteCare - Aggregator" `
    -Command "python .\Aggregator_Server\server\server_main.py --mode distributed --no-confirm-wait --startup-delay-seconds $StartupDelaySeconds"

Start-Sleep -Seconds 1

Start-DemoWindow `
    -Title "FedLiteCare - Hospital_A" `
    -Command "python .\Hospital_A\client\hospital_a_client.py federated-round"

Start-Sleep -Seconds 1

Start-DemoWindow `
    -Title "FedLiteCare - Hospital_B" `
    -Command "python .\Hospital_B\client\hospital_b_client.py federated-round"

Start-Sleep -Seconds 1

Start-DemoWindow `
    -Title "FedLiteCare - Hospital_C" `
    -Command "python .\Hospital_C\client\hospital_c_client.py federated-round"

Write-Host "FedLiteCare demo terminals launched."
Write-Host "Aggregator auto-start delay: $StartupDelaySeconds second(s)."
Write-Host "After the round finishes, review Demo_Outputs\demo_logs and Demo_Outputs\test_outputs."
