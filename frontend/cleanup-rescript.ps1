# PowerShell script to clean up ReScript files now that we've migrated to Scala.js

# Remove ReScript source files
Write-Host "Removing ReScript source files..."
Get-ChildItem -Path "./src" -Filter "*.res" -Recurse | Remove-Item -Force

# Remove ReScript generated files
Write-Host "Removing ReScript generated files..."
Get-ChildItem -Path "./src" -Filter "*.res.js" -Recurse | Remove-Item -Force

# Remove ReScript config files if they exist
Write-Host "Removing ReScript configuration files..."
if (Test-Path "./rescript.json") { Remove-Item -Path "./rescript.json" -Force }
if (Test-Path "./bsconfig.json") { Remove-Item -Path "./bsconfig.json" -Force }

# Clean up ReScript cache folders
Write-Host "Removing ReScript cache folders..."
if (Test-Path "./lib") { Remove-Item -Path "./lib" -Recurse -Force }
if (Test-Path "./.merlin") { Remove-Item -Path "./.merlin" -Force }

Write-Host "ReScript cleanup complete!"
