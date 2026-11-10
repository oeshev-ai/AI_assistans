# Snapshot Redis into .\backups\redis via redis-cli --rdb.
# Usage:
#   .\scripts\backup_redis.ps1
#   $env:REDIS_HOST='localhost'; .\scripts\backup_redis.ps1

$ErrorActionPreference = 'Stop'

$RedisHost = if ($env:REDIS_HOST) { $env:REDIS_HOST } else { '127.0.0.1' }
$RedisPort = if ($env:REDIS_PORT) { $env:REDIS_PORT } else { '6379' }
$BackupDir = if ($env:BACKUP_DIR) { $env:BACKUP_DIR } else { '.\backups\redis' }
$KeepDays = if ($env:KEEP_DAYS) { [int]$env:KEEP_DAYS } else { 7 }

New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$target = Join-Path $BackupDir "dump-$stamp.rdb"

$cli = Get-Command redis-cli -ErrorAction SilentlyContinue
if (-not $cli) {
    Write-Error 'redis-cli not found in PATH. Install Redis tools or run the shell script inside a redis container.'
}

Write-Host "Triggering BGSAVE on ${RedisHost}:${RedisPort}..."
& redis-cli -h $RedisHost -p $RedisPort BGSAVE | Out-Null
Start-Sleep -Seconds 2

Write-Host "Saving RDB to $target"
& redis-cli -h $RedisHost -p $RedisPort --rdb $target
if ($LASTEXITCODE -ne 0) {
    Write-Error "redis-cli --rdb failed with exit $LASTEXITCODE"
}

Get-ChildItem -Path $BackupDir -Filter 'dump-*.rdb' |
    Where-Object { $_.LastWriteTimeUtc -lt (Get-Date).ToUniversalTime().AddDays(-$KeepDays) } |
    Remove-Item -Force

Write-Host "Backup complete: $target"
Get-Item $target | Format-List Name, Length, LastWriteTimeUtc
