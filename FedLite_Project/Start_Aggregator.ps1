$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python .\Aggregator_Server\server\server_main.py --mode distributed
