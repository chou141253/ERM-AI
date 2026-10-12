param([Parameter(Mandatory=$true)][string]$Case, [Parameter(Mandatory=$true)][string]$Python)
# Offline checks only: GitHub, CLI and browser operations are replaced below.
$ErrorActionPreference = 'Stop'
$TestInterpreter = $Python
. (Join-Path $PSScriptRoot 'publish_erm_ai.ps1') -DefinitionsOnly
$WorkRoot = Join-Path $CodeRoot ('.publish-work/test-windows-' + [guid]::NewGuid().ToString('N'))
$SessionRoot = Join-Path $WorkRoot 'session'
New-Item -ItemType Directory -Path $SessionRoot -Force | Out-Null

if ($Case -eq 'native-exit') {
    $success = Test-Native $TestInterpreter @('-c', "print('visible progress')")
    if ($success -isnot [bool] -or -not $success) { throw 'Native success status was contaminated by stdout.' }
    $failure = Test-Native $TestInterpreter @('-c', "import sys; print('visible failure'); print('expected stderr',file=sys.stderr); sys.exit(7)")
    if ($failure -isnot [bool] -or $failure) { throw 'Native failure was incorrectly accepted.' }
    Write-Host 'Native status checks passed.'
    exit 0
}

if ($Case -eq 'source-list') {
    $files = @(Get-PublicFiles)
    if ($files.Count -lt 10) { throw 'Unexpectedly incomplete public source package.' }
    foreach ($file in $files) {
        if ($file.path -match '(^|/)(old|audit|runs|artifacts|smoke_output|\.venv|\.cache|\.publish-work)(/|$)|verification\.json|\.(pth|pt|safetensors|png|jpg|jpeg)$') {
            throw ('Private/generated path selected: ' + $file.path)
        }
    }
    if (-not @($files | Where-Object { $_.path -eq 'new/main.py' })) { throw 'Missing main entry point.' }
    Write-Host ('Public source list passed: ' + $files.Count + ' files.')
    exit 0
}

$script:FakeCreated = ($Case -eq 'wrong-fork')
$script:Calls = New-Object System.Collections.Generic.List[object]
function Get-Gh { return 'mock-gh-never-executed' }
function Test-Native { return $true }
function Invoke-Native([string]$Exe, [string[]]$Arguments) {
    $script:Calls.Add(@{ kind = 'native'; arguments = $Arguments })
    if ($Arguments[0] -eq 'repo' -and $Arguments[1] -eq 'fork') { $script:FakeCreated = $true }
}
function Start-Process { $script:Calls.Add(@{ kind = 'browser' }); return $null }
function Invoke-GhJson([string]$Method, [string]$Endpoint, $Body = $null, [switch]$Allow404) {
    $script:Calls.Add(@{ kind = 'api'; method = $Method; endpoint = $Endpoint; body = $Body })
    switch ($Endpoint) {
        'user' { return [pscustomobject]@{ login = 'publisher-test'; id = 101; email = 'private@example.test' } }
        'repos/ntuhophai/ERM-AI' { return [pscustomobject]@{ id = 202; default_branch = 'main' } }
        'repos/publisher-test/ERM-AI' {
            if (-not $script:FakeCreated) { return $null }
            $parentId = if ($Case -eq 'wrong-fork') { 999 } else { 202 }
            return [pscustomobject]@{ owner = @{ login = 'publisher-test' }; fork = $true;
                parent = @{ id = $parentId; full_name = 'ntuhophai/ERM-AI' }; permissions = @{ push = $true } }
        }
        'repos/ntuhophai/ERM-AI/git/ref/heads/main' { return [pscustomobject]@{ object = @{ sha = ('a' * 40) } } }
        { $_ -eq ('repos/ntuhophai/ERM-AI/git/commits/' + ('a' * 40)) } { return [pscustomobject]@{ sha = ('a' * 40); tree = @{ sha = ('b' * 40) } } }
        'repos/publisher-test/ERM-AI/git/blobs' { return [pscustomobject]@{ sha = ('c' * 40) } }
        'repos/publisher-test/ERM-AI/git/trees' { return [pscustomobject]@{ sha = ('d' * 40) } }
        'repos/publisher-test/ERM-AI/git/commits' { return [pscustomobject]@{ sha = ('e' * 40) } }
        'repos/publisher-test/ERM-AI/git/refs' { return [pscustomobject]@{ ref = $Body.ref } }
        default { throw ('Unmocked endpoint: ' + $Endpoint) }
    }
}
$payload = @([pscustomobject]@{ path = 'new/main.py'; bytes = $Utf8.GetBytes('print("public fixture")') })
if ($Case -eq 'wrong-fork') {
    $rejected = $false
    try { Publish-GitHub $payload } catch { $rejected = $true }
    if (-not $rejected) { throw 'An unrelated existing fork was accepted.' }
    if (@($script:Calls | Where-Object { $_.kind -eq 'api' -and $_.method -eq 'POST' }).Count) { throw 'Wrong fork received writes.' }
    Write-Host 'Wrong fork rejected before writes.'
    exit 0
}
if ($Case -ne 'github-flow') { throw 'Unknown offline test case.' }
Publish-GitHub $payload
$posts = @($script:Calls | Where-Object { $_.kind -eq 'api' -and $_.method -eq 'POST' })
if ($posts.Count -ne 4) { throw 'Unexpected GitHub mutation sequence.' }
foreach ($call in $posts) {
    if (-not $call.endpoint.StartsWith('repos/publisher-test/ERM-AI/')) { throw 'A write targeted upstream.' }
}
$treeCall = $posts | Where-Object { $_.endpoint.EndsWith('/trees') }
if ($treeCall.body.base_tree -ne ('b' * 40) -or $treeCall.body.tree[0].path -ne 'new/main.py') { throw 'Source tree was not based on upstream.' }
$commitCall = $posts | Where-Object { $_.endpoint.EndsWith('/commits') }
if ($commitCall.body.author.email -ne '101+publisher-test@users.noreply.github.com') { throw 'Private author email was used.' }
$refCall = $posts | Where-Object { $_.endpoint.EndsWith('/refs') }
if ($refCall.body.ref -notmatch '^refs/heads/publish/new-code-') { throw 'Default branch was targeted.' }
if (-not (Test-Path -LiteralPath (Join-Path $WorkRoot 'last-github.json'))) { throw 'Missing completion receipt.' }
Write-Host 'Mock GitHub fork/source branch flow passed. No remote commands were executed.'
