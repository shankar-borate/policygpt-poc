@echo off
setlocal

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root = (Get-Location).Path; " ^
  "$log = Join-Path $root 'app.log'; " ^
  "$pidFile = Join-Path $root 'app.pid'; " ^
  "if (Test-Path $pidFile) { " ^
  "  $existingPid = (Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1); " ^
  "  if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) { " ^
  "    Write-Host ('Policy GPT is already running with PID ' + $existingPid); " ^
  "    Write-Host ('Log: ' + $log); " ^
  "    exit 0 " ^
  "  } " ^
  "} " ^
  "if (-not (Test-Path $log)) { New-Item -ItemType File -Path $log | Out-Null } " ^
  "$command = 'Set-Location -LiteralPath ''' + $root + '''; python app.py *>> ''' + $log + ''''; " ^
  "$process = Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-Command',$command -WindowStyle Hidden -PassThru; " ^
  "Set-Content -LiteralPath $pidFile -Value $process.Id; " ^
  "Write-Host ('Policy GPT started in background.'); " ^
  "Write-Host ('PID: ' + $process.Id); " ^
  "Write-Host ('Log: ' + $log)"

endlocal
