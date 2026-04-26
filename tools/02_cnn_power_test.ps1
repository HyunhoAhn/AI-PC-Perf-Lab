param(
    [string]$RunId = "cnn_power_test",
    [int]$AttemptsPerCase = 10,
    [int]$CooldownSeconds = 30,
    [string]$InputShape = "3x224x224",
    [int]$Batch = 1,
    [int]$Warmup = 10,
    [int]$Repeat = 20000,
    [string[]]$Tools = @("uprof", "xrt"),
    [string[]]$Cases = @(),
    [string]$PythonExe = "python",
    [string]$SharedVaipCacheDir = "results/raw/cnn_power_test",
    [switch]$SkipCaptureEnv
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$powerScript = Join-Path $repoRoot "src\CNNs\02_power_test.py"
if (-not (Test-Path $powerScript)) {
    throw "Power test script not found: $powerScript"
}

if ($AttemptsPerCase -le 0) {
    throw "-AttemptsPerCase must be positive."
}

if ($CooldownSeconds -lt 0) {
    throw "-CooldownSeconds must be zero or positive."
}

if ($Batch -le 0) {
    throw "-Batch must be positive."
}

if ($Warmup -lt 0) {
    throw "-Warmup must be zero or positive."
}

if ($Repeat -le 0) {
    throw "-Repeat must be positive."
}

$normalizedTools = @($Tools | ForEach-Object { $_.ToLowerInvariant() } | Select-Object -Unique)
$validTools = @("uprof", "xrt")
$invalidTools = @($normalizedTools | Where-Object { $_ -notin $validTools })
if ($invalidTools.Count -gt 0) {
    throw "Unsupported tool(s): $($invalidTools -join ', ')"
}

$caseMatrix = @(
    @{
        CaseName = "fp32_cpu"
        ModelPath = "models/resnet50.onnx"
        Device = "cpu"
        DisableFallback = $false
        VaipCacheKey = $null
    }
    @{
        CaseName = "fp32_npu"
        ModelPath = "models/resnet50.onnx"
        Device = "npu"
        DisableFallback = $true
        VaipCacheKey = "resnet50_FP32"
    }
    @{
        CaseName = "fp32_igpu"
        ModelPath = "models/resnet50.onnx"
        Device = "igpu"
        DisableFallback = $true
        VaipCacheKey = $null
    }
    @{
        CaseName = "int8_cpu"
        ModelPath = "models/resnet50_A8W8.onnx"
        Device = "cpu"
        DisableFallback = $false
        VaipCacheKey = $null
    }
    @{
        CaseName = "int8_npu"
        ModelPath = "models/resnet50_A8W8.onnx"
        Device = "npu"
        DisableFallback = $true
        VaipCacheKey = "resnet50_A8W8"
    }
    @{
        CaseName = "int8_igpu"
        ModelPath = "models/resnet50_A8W8.onnx"
        Device = "igpu"
        DisableFallback = $true
        VaipCacheKey = $null
    }
)

$selectedCases = if ($Cases.Count -gt 0) {
    $wanted = @($Cases | ForEach-Object { $_.ToLowerInvariant() })
    $missingCases = @($wanted | Where-Object { $_ -notin $caseMatrix.CaseName })
    if ($missingCases.Count -gt 0) {
        throw "Unsupported case(s): $($missingCases -join ', ')"
    }
    @($caseMatrix | Where-Object { $_.CaseName -in $wanted })
} else {
    $caseMatrix
}

function Get-TelemetryToolsForCase {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Case
    )

    $caseTools = @("uprof")
    if ($Case.Device -eq "npu") {
        $caseTools += "xrt"
    }

    return @($caseTools | Where-Object { $_ -in $normalizedTools })
}

function Invoke-PowerCase {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Case,

        [Parameter(Mandatory = $true)]
        [string]$Tool,

        [Parameter(Mandatory = $true)]
        [int]$AttemptIndex
    )

    Write-Host ""
    Write-Host "===== $RunId :: $($Case.CaseName) :: $Tool :: attempt $AttemptIndex/$AttemptsPerCase ====="

    $command = @(
        "src/CNNs/02_power_test.py"
        "--run-id", $RunId
        "--case-name", $Case.CaseName
        "--attempt-index", "$AttemptIndex"
        "--telemetry-tool", $Tool
        "--model-path", $Case.ModelPath
        "--device", $Case.Device
        "--input-shape", $InputShape
        "--batch", "$Batch"
        "--warmup", "$Warmup"
        "--repeat", "$Repeat"
    )

    if ($Case.DisableFallback) {
        $command += "--disable-fallback"
    }

    if ($Case.Device -eq "npu") {
        $command += @(
            "--shared-vaip-cache-dir", $SharedVaipCacheDir
            "--vaip-cache-key", $Case.VaipCacheKey
        )
    }

    & $PythonExe "tools/run_capture.py" --run-id $RunId -- $PythonExe @command
}

if (-not $SkipCaptureEnv) {
    Write-Host "===== capture_env: $RunId ====="
    & $PythonExe "tools/capture_env.py" --run-id $RunId
}

$plannedAttempts = 0
foreach ($case in $selectedCases) {
    $plannedAttempts += @(Get-TelemetryToolsForCase -Case $case).Count * $AttemptsPerCase
}
Write-Host "===== planned measured attempts: $plannedAttempts ====="

$hasStartedAnyAttempt = $false

for ($attemptIndex = 1; $attemptIndex -le $AttemptsPerCase; $attemptIndex++) {
    foreach ($case in $selectedCases) {
        $caseTools = @(Get-TelemetryToolsForCase -Case $case)
        foreach ($tool in $caseTools) {
            if ($hasStartedAnyAttempt -and $CooldownSeconds -gt 0) {
                Write-Host ""
                Write-Host "===== cooldown: waiting $CooldownSeconds seconds ====="
                Start-Sleep -Seconds $CooldownSeconds
            }

            Invoke-PowerCase -Case $case -Tool $tool -AttemptIndex $attemptIndex
            $hasStartedAnyAttempt = $true
        }
    }
}
