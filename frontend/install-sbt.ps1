# Script to install SBT (Scala Build Tool) for Windows

# Create directory for SBT installation if it doesn't exist
$installDir = "$env:USERPROFILE\sbt"
if (!(Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force
}

# Define SBT version and download URL
$sbtVersion = "1.9.7"
$sbtUrl = "https://github.com/sbt/sbt/releases/download/v$sbtVersion/sbt-$sbtVersion.zip"
$zipFile = "$env:TEMP\sbt.zip"

# Download SBT
Write-Host "Downloading SBT $sbtVersion..."
Invoke-WebRequest -Uri $sbtUrl -OutFile $zipFile

# Extract SBT
Write-Host "Extracting SBT..."
Expand-Archive -Path $zipFile -DestinationPath $installDir -Force

# Add SBT to PATH temporarily for this session
$env:Path = "$installDir\sbt\bin;$env:Path"

# Test SBT
Write-Host "Testing SBT installation..."
sbt sbtVersion

Write-Host "SBT installation complete! Please add $installDir\sbt\bin to your system PATH for permanent access."

# Create or update .sbtopts file for memory settings
$sbtopts = @"
-J-Xmx2G
-J-Xms512M
-J-XX:+UseG1GC
"@
Set-Content -Path ".sbtopts" -Value $sbtopts

Write-Host "SBT memory settings configured in .sbtopts file."
