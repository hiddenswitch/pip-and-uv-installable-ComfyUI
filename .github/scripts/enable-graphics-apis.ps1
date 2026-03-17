function CopyToSystem32($sourceDirectory, $filenames, $rename) {
    foreach ($filename in $filenames) {
        $source = Join-Path $sourceDirectory $filename
        $destinationName = $filename
        if ($rename -and $rename[$filename]) {
            $destinationName = $rename[$filename]
        }
        $destination = Join-Path "C:\Windows\System32" $destinationName
        Write-Host "Copying $source to $destination"
        try {
            Copy-Item -Path $source -Destination $destination -ErrorAction Stop
        } catch {
            Write-Host "Warning: failed to copy file $filename" -ForegroundColor Yellow
        }
    }
}

$nvidiaSentinelFile = Get-ChildItem "C:\Windows\System32\HostDriverStore\FileRepository\nv*.inf_amd64_*\nvapi64.dll" -ErrorAction SilentlyContinue
if (-not $nvidiaSentinelFile) {
    Write-Host "No NVIDIA HostDriverStore files found; nothing to copy."
    exit 0
}

$nvidiaDirectory = Split-Path $nvidiaSentinelFile[0].VersionInfo.FileName
Write-Host "Found NVIDIA Display Driver directory: $nvidiaDirectory"

Write-Host "`nEnabling NVIDIA NVAPI support:"
CopyToSystem32 -sourceDirectory $nvidiaDirectory -filenames @("nvapi64.dll", "nvml.dll")

Write-Host "`nEnabling NVIDIA NVENC support:"
CopyToSystem32 -sourceDirectory $nvidiaDirectory -filenames @("nvEncodeAPI64.dll", "nvEncMFTH264x.dll", "nvEncMFThevcx.dll")

Write-Host "`nEnabling NVIDIA CUVID/NVDEC support:"
CopyToSystem32 -sourceDirectory $nvidiaDirectory -filenames @("nvcuvid64.dll", "nvDecMFTMjpeg.dll", "nvDecMFTMjpegx.dll") -rename @{"nvcuvid64.dll" = "nvcuvid.dll"}

Write-Host "`nEnabling NVIDIA CUDA support:"
CopyToSystem32 -sourceDirectory $nvidiaDirectory -filenames @("nvcuda64.dll", "nvcuda_loader64.dll", "nvptxJitCompiler64.dll") -rename @{"nvcuda_loader64.dll" = "nvcuda.dll"}
