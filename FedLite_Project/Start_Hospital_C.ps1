$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python .\Hospital_C\client\hospital_c_client.py federated-round
