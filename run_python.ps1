param(
    [Parameter(Position = 0)]
    [string]$TargetPath,
    [int]$Parallel,
    [int]$TranscribeParallel,
    [int]$TranslateChunkSize,
    [switch]$FastTranscribe,
    [switch]$SequentialTranscribe,
    [switch]$TranscribeOnly,
    [string]$InitialPrompt,
    [switch]$NoAsmrPrompt,
    [switch]$DryRun
)

# Use the shared launcher for conda resolution, working directory and defaults.
& (Join-Path $PSScriptRoot "run.ps1") @PSBoundParameters
exit $LASTEXITCODE
