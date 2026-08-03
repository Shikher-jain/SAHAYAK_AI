$projectPath = $PWD.Path

Write-Host "Starting backend..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location -LiteralPath '$projectPath'; .\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --reload-dir backend --reload-dir frontend"
)

Write-Host "Starting frontend..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location -LiteralPath '$projectPath'; .\.venv\Scripts\python.exe -m streamlit run .\frontend\app.py"
)