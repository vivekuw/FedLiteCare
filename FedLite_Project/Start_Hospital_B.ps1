$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python .\Hospital_B\client\hospital_b_client.py federated-round
