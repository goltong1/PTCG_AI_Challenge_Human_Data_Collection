$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $ProjectRoot "runtime"
$PythonRoot = Join-Path $RuntimeRoot "python"
$ReadyMarker = Join-Path $RuntimeRoot ".cabt-ready-v2"
$PythonVersion = "3.11.9"
$PythonArchive = Join-Path $RuntimeRoot "python-embed.zip"
$GetPipPath = Join-Path $RuntimeRoot "get-pip.py"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"

function Download-File {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    $lastError = $null
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Write-Host "[CABT] Downloading $Url (attempt $attempt/3)"
            Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Destination
            return
        }
        catch {
            $lastError = $_
            Start-Sleep -Seconds 2
        }
    }
    throw $lastError
}

Write-Host ""
Write-Host "CABT portable runtime setup"
Write-Host "This runs once. No system-wide Python installation is created."
Write-Host ""

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
if (Test-Path $PythonRoot) {
    Remove-Item -Recurse -Force $PythonRoot
}
New-Item -ItemType Directory -Force -Path $PythonRoot | Out-Null

try {
    Download-File -Url $PythonUrl -Destination $PythonArchive
    Write-Host "[CABT] Extracting portable Python $PythonVersion..."
    Expand-Archive -Path $PythonArchive -DestinationPath $PythonRoot -Force

    $PthFile = Join-Path $PythonRoot "python311._pth"
    @(
        "python311.zip"
        "."
        "Lib\site-packages"
        "..\..\app"
        "..\.."
        "import site"
    ) | Set-Content -Path $PthFile -Encoding ASCII

    Download-File -Url $GetPipUrl -Destination $GetPipPath
    $PythonExe = Join-Path $PythonRoot "python.exe"
    Write-Host "[CABT] Preparing the bundled package runtime..."
    & $PythonExe $GetPipPath --disable-pip-version-check --no-warn-script-location
    if ($LASTEXITCODE -ne 0) {
        throw "pip bootstrap failed with exit code $LASTEXITCODE"
    }
    & $PythonExe -m pip install --disable-pip-version-check --no-warn-script-location --only-binary=:all: "numpy==2.0.2"
    if ($LASTEXITCODE -ne 0) {
        throw "NumPy setup failed with exit code $LASTEXITCODE"
    }
    & $PythonExe -c "import ctypes, json, numpy; print('[CABT] Runtime check OK - NumPy', numpy.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "runtime verification failed with exit code $LASTEXITCODE"
    }

    "CABT runtime v2 - Python $PythonVersion" | Set-Content -Path $ReadyMarker -Encoding ASCII
    Write-Host ""
    Write-Host "[CABT] Portable runtime is ready."
}
finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $PythonArchive
    Remove-Item -Force -ErrorAction SilentlyContinue $GetPipPath
}
