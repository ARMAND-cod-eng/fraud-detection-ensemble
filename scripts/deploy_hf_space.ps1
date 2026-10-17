<#
.SYNOPSIS
  Publish the FraudShield demo to a Hugging Face Space (Docker SDK).

.DESCRIPTION
  Prerequisites (one time, in your own account):
    1. Create a Space at https://huggingface.co/new-space  (SDK: Docker, template: Blank, hardware: CPU basic).
    2. Create an access token with write permission at https://huggingface.co/settings/tokens.
  When git asks for credentials during the push, enter your HF username and that token as the
  password. Git's credential manager stores it; this script never reads or prints it.

  The Space receives exactly the committed code (git archive of HEAD), plus deploy/hf-space/Dockerfile
  and deploy/hf-space/README.md at its root. Hugging Face then builds the image, which downloads the
  dataset and trains the model.

.EXAMPLE
  .\scripts\deploy_hf_space.ps1 -SpaceId your-hf-username/fraudshield
#>
param([Parameter(Mandatory = $true)][string]$SpaceId)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$stage = Join-Path $env:TEMP "fraudshield-hf-space"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }

git clone "https://huggingface.co/spaces/$SpaceId" $stage
if ($LASTEXITCODE -ne 0) { throw "could not clone the Space; create it first (see the help above)" }

Push-Location $repo
$sha = (git rev-parse --short HEAD).Trim()
$archive = Join-Path $env:TEMP "fraudshield-src.tar"
git archive --format=tar -o $archive HEAD src api configs requirements dashboard pyproject.toml LICENSE
Pop-Location
tar -xf $archive -C $stage
Remove-Item $archive
Copy-Item (Join-Path $repo "deploy\hf-space\Dockerfile") (Join-Path $stage "Dockerfile") -Force
Copy-Item (Join-Path $repo "deploy\hf-space\README.md") (Join-Path $stage "README.md") -Force

Push-Location $stage
git add -A
git commit -m "Deploy FraudShield $sha"
git push
Pop-Location
Write-Host "Pushed $sha. Follow the build at https://huggingface.co/spaces/$SpaceId (logs: 'Building')."
