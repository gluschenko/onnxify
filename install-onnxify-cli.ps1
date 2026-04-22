[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$Configuration = "Release",
    [string]$PackOutput = "artifacts\nupkgs\onnxify-cli"
)

$ErrorActionPreference = "Stop"

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Command
    )

    & $Command[0] $Command[1..($Command.Length - 1)]

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

function Get-ProjectMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectPath
    )

    [xml]$projectXml = Get-Content -LiteralPath $ProjectPath
    $propertyGroups = @($projectXml.Project.PropertyGroup)

    $packageId = ($propertyGroups | ForEach-Object { $_.PackageId } | Where-Object { $_ } | Select-Object -First 1)
    if (-not $packageId) {
        $packageId = [System.IO.Path]::GetFileNameWithoutExtension($ProjectPath)
    }

    $packageVersion = ($propertyGroups | ForEach-Object { $_.PackageVersion } | Where-Object { $_ } | Select-Object -First 1)
    if (-not $packageVersion) {
        $packageVersion = ($propertyGroups | ForEach-Object { $_.Version } | Where-Object { $_ } | Select-Object -First 1)
    }
    if (-not $packageVersion) {
        $versionPrefix = ($propertyGroups | ForEach-Object { $_.VersionPrefix } | Where-Object { $_ } | Select-Object -First 1)
        $versionSuffix = ($propertyGroups | ForEach-Object { $_.VersionSuffix } | Where-Object { $_ } | Select-Object -First 1)
        if ($versionPrefix) {
            $packageVersion = if ($versionSuffix) { "$versionPrefix-$versionSuffix" } else { $versionPrefix }
        }
    }

    if (-not $packageVersion) {
        throw "Could not determine package version from $ProjectPath"
    }

    return @{
        PackageId = $packageId
        PackageVersion = $packageVersion
    }
}

function Test-GlobalToolInstalled {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PackageId
    )

    $toolList = & dotnet tool list --global
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to query globally installed dotnet tools."
    }

    foreach ($line in $toolList) {
        if ($line -match "^\s*$([regex]::Escape($PackageId))\s+") {
            return $true
        }
    }

    return $false
}

$projectPath = Join-Path $PSScriptRoot "src\Onnxify.CLI\Onnxify.CLI.csproj"
if (-not (Test-Path -LiteralPath $projectPath -PathType Leaf)) {
    throw "Project file was not found: $projectPath"
}

if (-not [System.IO.Path]::IsPathRooted($PackOutput)) {
    $PackOutput = Join-Path $PSScriptRoot $PackOutput
}

$metadata = Get-ProjectMetadata -ProjectPath $projectPath
$packageId = $metadata.PackageId
$packageVersion = $metadata.PackageVersion
$nupkgPath = Join-Path $PackOutput "$packageId.$packageVersion.nupkg"

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    throw "The 'dotnet' command was not found in PATH."
}

if (-not (Test-Path -LiteralPath $PackOutput -PathType Container)) {
    if ($PSCmdlet.ShouldProcess($PackOutput, "Create package output directory")) {
        New-Item -ItemType Directory -Path $PackOutput -Force | Out-Null
    }
}

if ($PSCmdlet.ShouldProcess($projectPath, "Pack $packageId $packageVersion")) {
    Invoke-ExternalCommand -Command @(
        "dotnet",
        "pack",
        $projectPath,
        "--configuration",
        $Configuration,
        "--output",
        $PackOutput,
        "--nologo"
    )
}

if ($WhatIfPreference) {
    Write-Host ""
    Write-Host "WhatIf complete."
    return
}

if (-not (Test-Path -LiteralPath $nupkgPath -PathType Leaf)) {
    throw "Packed tool package was not found: $nupkgPath"
}

$isInstalled = Test-GlobalToolInstalled -PackageId $packageId

if ($isInstalled) {
    if ($PSCmdlet.ShouldProcess($packageId, "Update global dotnet tool from local package source")) {
        Invoke-ExternalCommand -Command @(
            "dotnet",
            "tool",
            "update",
            $packageId,
            "--global",
            "--add-source",
            $PackOutput,
            "--version",
            $packageVersion,
            "--ignore-failed-sources"
        )
    }
}
else {
    if ($PSCmdlet.ShouldProcess($packageId, "Install global dotnet tool from local package source")) {
        Invoke-ExternalCommand -Command @(
            "dotnet",
            "tool",
            "install",
            $packageId,
            "--global",
            "--add-source",
            $PackOutput,
            "--version",
            $packageVersion,
            "--ignore-failed-sources"
        )
    }
}

Write-Host ""
Write-Host "Packed package: $nupkgPath"
Write-Host "Tool command: onnxify"
