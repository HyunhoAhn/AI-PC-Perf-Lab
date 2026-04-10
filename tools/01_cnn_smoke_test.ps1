$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$runId = "cnn_smoke_test"
$pythonExe = "python"
$smokeScript = Join-Path $repoRoot "src\CNNs\01_cnn_smoke_test.py"

if (-not (Test-Path $smokeScript)) {
    throw "Smoke test script not found: $smokeScript"
}

$commonArgs = @(
    "--input-shape", "3x224x224"
    "--batch", "1"
    "--repeat", "50"
    "--warmup", "10"
)

$coolDownSeconds = 5
$noProfileOuterLoop = 10
$script:HasStartedSmokeProcess = $false

$models = @(
    ".\models\resnet50.onnx"
    ".\models\resnet50_A8W8.onnx"
    ".\models\resnet50_FP16.onnx"
)

$devices = @(
    @{ Name = "cpu"; DisableFallback = $false }
    @{ Name = "npu"; DisableFallback = $true }
    @{ Name = "igpu"; DisableFallback = $true }
)

function Invoke-CnnSmokeCase {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ModelPath,

        [Parameter(Mandatory = $true)]
        [string]$Device,

        [Parameter(Mandatory = $true)]
        [bool]$DisableFallback,

        [Parameter(Mandatory = $true)]
        [string]$VariantName,

        [Parameter(Mandatory = $true)]
        [bool]$EnableProfile
    )

    if ($script:HasStartedSmokeProcess -and $coolDownSeconds -gt 0) {
        Write-Host ""
        Write-Host "===== cooldown: waiting $coolDownSeconds seconds ====="
        Start-Sleep -Seconds $coolDownSeconds
    }

    Write-Host ""
    Write-Host "===== $runId :: $ModelPath :: $Device :: $VariantName ====="

    $modelName = [System.IO.Path]::GetFileNameWithoutExtension($ModelPath)
    $command = @(
        $smokeScript
        "--model-path", $ModelPath
        "--device", $Device
    ) + $commonArgs

    if ($DisableFallback) {
        $command += "--disable-fallback"
    }

    if ($EnableProfile) {
        $command += @("--profile-out", "results/raw/$runId")
    }

    if ($Device -eq "npu") {
        $command += @(
            "--vaip-cache-dir", "results/raw/$runId"
            "--vaip-cache-key", $modelName
            "--clear-vaip-cache"
        )
    }

    & $pythonExe "tools/run_capture.py" --run-id $runId -- $pythonExe @command
    $script:HasStartedSmokeProcess = $true
}

Write-Host "===== capture_env: $runId ====="
& $pythonExe "tools/capture_env.py" --run-id $runId

for ($outer = 1; $outer -le $noProfileOuterLoop; $outer++) {
    Write-Host ""
    Write-Host "===== no_profile iteration $outer/$noProfileOuterLoop ====="

    foreach ($modelPath in $models) {
        foreach ($device in $devices) {
            Invoke-CnnSmokeCase `
                -ModelPath $modelPath `
                -Device $device.Name `
                -DisableFallback $device.DisableFallback `
                -VariantName "no_profile" `
                -EnableProfile $false
        }
    }
}

foreach ($modelPath in $models) {
    foreach ($device in $devices) {
        Invoke-CnnSmokeCase `
            -ModelPath $modelPath `
            -Device $device.Name `
            -DisableFallback $device.DisableFallback `
            -VariantName "profile" `
            -EnableProfile $true
    }
}
