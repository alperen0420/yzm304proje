$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Sanal ortam bulunamadi. Once .\\bootstrap.ps1 calistirin."
}

& ".\.venv\Scripts\python.exe" -m src.run_all

