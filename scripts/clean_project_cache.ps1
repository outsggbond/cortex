param(
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPattern = "[\\/]\.venv[\\/]"
$cacheDirs = @("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")

Write-Host "Cleaning caches under $root (excluding .venv)..."
if ($DryRun) {
  Write-Host "Dry run enabled, no files will be deleted."
}

$stats = @{
  removed_dirs  = 0
  removed_files = 0
}

$dirs = Get-ChildItem -Path $root -Recurse -Directory -Force -ErrorAction SilentlyContinue |
  Where-Object { ($cacheDirs -contains $_.Name) -and ($_.FullName -notmatch $venvPattern) }

foreach ($dir in $dirs) {
  if ($DryRun) {
    Write-Host "[DRY] Remove directory: $($dir.FullName)"
    $stats.removed_dirs += 1
    continue
  }
  Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
  $stats.removed_dirs += 1
}

$files = Get-ChildItem -Path $root -Recurse -File -Force -ErrorAction SilentlyContinue |
  Where-Object {
    $_.FullName -notmatch $venvPattern -and (
      $_.Extension -in @(".pyc", ".pyo") -or
      $_.Name -eq ".coverage" -or
      $_.Name.StartsWith(".coverage.")
    )
  }

foreach ($file in $files) {
  if ($DryRun) {
    Write-Host "[DRY] Remove file: $($file.FullName)"
    $stats.removed_files += 1
    continue
  }
  Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
  $stats.removed_files += 1
}

Write-Host ("Cache cleanup done. Directories: {0}, files: {1}." -f $stats.removed_dirs, $stats.removed_files)
