$ErrorActionPreference='Stop'
$out='D:\Simone\proteinGen_data\thesis_storage_inventory_v1'
$env:TEMP=$out; $env:TMP=$out
$start=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0
$drive=[System.IO.DriveInfo]::new('D')
$before=@{total=$drive.TotalSize;free=$drive.AvailableFreeSpace}
$roots=@('D:\Simone\proteinGen','D:\Users\Simone Stocco\proteinGen_audits','D:\Simone\proteinGen_data\sequence_foundation\uniref50_2026_03_e012_v2','D:\Simone\proteinGen_data\sequence_foundation\e012_large_corpus_warmstart_20m_20261013_v1','D:\Simone\proteinGen_data\thesis_pdb_sequence_audit_v1','D:\Simone\proteinGen_data\thesis_pdb_sequence_audit_v2')
$dirs=@{}; $large=[System.Collections.Generic.List[object]]::new(); $errors=[System.Collections.Generic.List[string]]::new(); $count=0; $last=$start
function LinuxPath([string]$p) {return '/mnt/d'+$p.Substring(2).Replace('\','/')}
foreach($root in $roots) {
 Write-Output "SCAN $root"
 $stack=[System.Collections.Generic.Stack[string]]::new(); $stack.Push($root)
 while($stack.Count -gt 0) {
  $base=$stack.Pop(); $row=@{bytes=[long]0;files=0;symlinks=0}
  try {
   foreach($f in [System.IO.DirectoryInfo]::new($base).EnumerateFileSystemInfos()) {
    if(($f.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {$row.symlinks++; continue}
    if(($f.Attributes -band [System.IO.FileAttributes]::Directory) -ne 0) {$stack.Push($f.FullName); continue}
    $n=$f.Length; $row.bytes+=$n; $row.files++; $count++
    if($n -ge 104857600) {$large.Add(@{path=(LinuxPath $f.FullName);bytes=$n;mtime_utc=$f.LastWriteTimeUtc.ToString('o')})}
   }
  } catch {$errors.Add("${base}: $($_.Exception.Message)")}
  $dirs[(LinuxPath $base)]=$row
  $now=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0
  if($now-$last -gt 25) {Write-Output "PROGRESS files=$count"; $last=$now}
 }
}
foreach($p in @($dirs.Keys | Sort-Object Length -Descending)) {
 $parent=$p.Substring(0,$p.LastIndexOf('/'))
 if($dirs.ContainsKey($parent)) {foreach($k in @('bytes','files','symlinks')) {$dirs[$parent][$k]+=$dirs[$p][$k]}}
}
$rootRows=@(); foreach($r in $roots) {$p=LinuxPath $r; $rootRows+=@{path=$p;bytes=$dirs[$p].bytes;files=$dirs[$p].files;symlinks=$dirs[$p].symlinks}}
$data=@{started_unix=$start;finished_unix=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0;disk_before=$before;disk_after=@{total=$drive.TotalSize;free=$drive.AvailableFreeSpace};roots=$rootRows;directories=$dirs;largest_files=@($large | Sort-Object { $_['bytes'] } -Descending | Select-Object -First 200);regular_files=$count;errors=@($errors);method='Sequential native NTFS DirectoryInfo enumeration; apparent file lengths; excludes reparse points; no content reads or hashes'}
[System.IO.File]::WriteAllText("$out\INVENTORY.json",($data | ConvertTo-Json -Depth 8 -Compress))
Write-Output (@{regular_files=$count;errors=@($errors);seconds=$data.finished_unix-$start} | ConvertTo-Json -Depth 3)
