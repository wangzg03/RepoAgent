param(
    [string]$Python = "",
    [switch]$PrepareOnly,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$AgentArguments
)
$ErrorActionPreference = "Stop"
if (-not $Python) { $Python = $env:CORECODER_PYTHON }
if (-not $Python) {
    $candidate = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate) { $Python = $candidate }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
Push-Location $PSScriptRoot
try {
    $parameters = @("-X", "utf8", "-B", (Join-Path $PSScriptRoot "run.py"))
    if (-not $PrepareOnly) { $parameters += "--run" }
    $parameters += $AgentArguments
    & $Python @parameters
    $result = $LASTEXITCODE
}
finally { Pop-Location }
exit $result
