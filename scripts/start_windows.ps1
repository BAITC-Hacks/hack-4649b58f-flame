param([string]$OllamaPath = "")

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create .venv and install requirements-ui.txt and requirements-audio.txt first (see README).'
}
if (-not $OllamaPath) {
    $ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaCommand) { $OllamaPath = $ollamaCommand.Source }
}
if (-not $OllamaPath -or -not (Test-Path -LiteralPath $OllamaPath)) {
    throw 'Install Ollama or pass -OllamaPath C:\path\to\ollama.exe'
}
$OllamaPath = (Resolve-Path -LiteralPath $OllamaPath).Path
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = '1'
if (-not $env:OLLAMA_MODEL) { $env:OLLAMA_MODEL = 'qwen2.5:3b-instruct-q4_K_M' }
if (-not $env:OLLAMA_NUM_CTX) { $env:OLLAMA_NUM_CTX = '8192' }
if (-not $env:OLLAMA_TIMEOUT_SECONDS) { $env:OLLAMA_TIMEOUT_SECONDS = '180' }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $projectRoot 'models\huggingface' }
$env:HF_HUB_DISABLE_TELEMETRY = '1'

function Test-LocalOllama {
    try {
        $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2
        return $true
    } catch { return $false }
}

if (-not (Test-LocalOllama)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot 'output') | Out-Null
    Start-Process -FilePath $OllamaPath -ArgumentList 'serve' -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $projectRoot 'output\ollama-out.log') `
        -RedirectStandardError (Join-Path $projectRoot 'output\ollama-error.log') | Out-Null
    for ($attempt = 0; $attempt -lt 20 -and -not (Test-LocalOllama); $attempt++) {
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-LocalOllama)) { throw 'Ollama did not start. Check output\ollama-error.log.' }
}
& $OllamaPath pull $env:OLLAMA_MODEL
if ($LASTEXITCODE -ne 0) { throw 'Ollama model download failed.' }
& $pythonPath -m streamlit run streamlit_app.py --server.address 127.0.0.1
if ($LASTEXITCODE -ne 0) { throw 'Streamlit stopped with an error.' }
