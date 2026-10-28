param(
    [Parameter(Position = 0)]
    [string]$TargetPath = "W:\github\VoiceTransl\input",
    [int]$Parallel = 5,
    [int]$TranscribeParallel = 1,
    [int]$TranslateChunkSize = 20,
    [switch]$FastTranscribe,
    [switch]$SequentialTranscribe,
    [switch]$TranscribeOnly,
    [string]$InitialPrompt = "",
    [switch]$NoAsmrPrompt,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
if ($FastTranscribe -and $SequentialTranscribe) {
    throw "Choose either -FastTranscribe or -SequentialTranscribe."
}

# Keep Japanese / Chinese paths and SRT text stable in PowerShell output.
chcp 65001 | Out-Null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location -LiteralPath $PSScriptRoot

function Resolve-CondaExe {
    $stableCandidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\condabin\conda.bat"),
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe")
    )

    foreach ($candidate in $stableCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCommand) {
        return $condaCommand.Source
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE ".conda\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "conda executable not found. Please install conda or add it to PATH."
}

$condaExe = Resolve-CondaExe
$inputPath = $TargetPath
if (-not (Test-Path -LiteralPath $inputPath)) {
    throw "Input path does not exist: $inputPath"
}
$inputPath = (Resolve-Path -LiteralPath $inputPath).Path

$modeFlag = if ($DryRun) { "--dry-run" } else { "--execute" }

$arguments = @(
    "scripts/asmr_transcribe_srt_batch.py",
    "--path", $inputPath,
    $modeFlag,
    "--parallel", "$Parallel",
    "--transcribe_parallel", "$TranscribeParallel",
    "--translate_chunk_size", "$TranslateChunkSize"
)

if ($FastTranscribe) {
    $arguments += "--fast_transcribe"
}
if ($SequentialTranscribe) {
    $arguments += "--sequential_transcribe"
}
if ($TranscribeOnly) {
    $arguments += "--transcribe_only"
}
if ($NoAsmrPrompt) {
    $arguments += "--no_asmr_prompt"
} elseif ($InitialPrompt -ne "") {
    $arguments += @("--initial_prompt", $InitialPrompt)
}

Write-Host "Running ASMR transcription pipeline..."
Write-Host "Project: $PSScriptRoot"
Write-Host "Input path: $inputPath"
Write-Host "Mode: $modeFlag"
Write-Host "Parallel: $Parallel  |  Transcribe parallel: $TranscribeParallel"
Write-Host "Translate chunk size: $TranslateChunkSize"
Write-Host "Transcription preset: $(if ($SequentialTranscribe) { 'sequential' } else { 'batched (batch_size=8, sentence=True, chunk_length=20)' })"
Write-Host "Conda: $condaExe"
Write-Host ""

& $condaExe run --no-capture-output -n asmr python -u @arguments
exit $LASTEXITCODE
