# Скачивает MNIST в папку data\ (Windows PowerShell).
# Запуск из корня проекта:  powershell -ExecutionPolicy Bypass -File scripts\download_mnist.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $root "data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

$base = "https://ossci-datasets.s3.amazonaws.com/mnist"
$files = @("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
           "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")

foreach ($f in $files) {
    $target = Join-Path $dataDir $f
    if (Test-Path $target) {
        Write-Host "$target уже есть"
        continue
    }
    $gz = "$target.gz"
    Write-Host "скачиваю $f ..."
    Invoke-WebRequest -Uri "$base/$f.gz" -OutFile $gz

    # распаковка .gz средствами .NET (без сторонних программ)
    $in = [System.IO.File]::OpenRead($gz)
    $out = [System.IO.File]::Create($target)
    $gzip = New-Object System.IO.Compression.GZipStream($in, [System.IO.Compression.CompressionMode]::Decompress)
    $gzip.CopyTo($out)
    $gzip.Dispose(); $out.Dispose(); $in.Dispose()
    Remove-Item $gz
}
Write-Host "готово: MNIST лежит в $dataDir"
