$ErrorActionPreference = 'Stop'
$out = 'D:\Simone\proteinGen\storage_audit_20261011'
$env:TEMP=$out
$env:TMP=$out
$start=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0
$sizes=@{}
$large=[System.Collections.Generic.List[object]]::new()
$special=[System.Collections.Generic.List[object]]::new()
$checkpoints=[System.Collections.Generic.List[object]]::new()
$errors=[System.Collections.Generic.List[string]]::new()
function LinuxPath([string]$p) { return '/mnt/d'+$p.Substring(2).Replace('\','/') }
function Owner([string]$p) {
  foreach($e in @('e012','e015','e007','e014')) { if($p.ToLower().Contains($e)) {return $e.ToUpper()} }
  if($p.Contains('sequence_foundation')) {return 'E012'}
  if($p.Contains('proteinGen_data')) {return 'shared dataset'}
  return 'unknown'
}
$stack=[System.Collections.Generic.Stack[string]]::new()
foreach($root in @('D:\Simone\proteinGen','D:\Simone\proteinGen_data')) {$stack.Push($root)}
while($stack.Count -gt 0) {
 $base=$stack.Pop()
 if($base -eq $out) {continue}
 $total=[long]0;$count=0
 try {
  foreach($p in [System.IO.Directory]::EnumerateFileSystemEntries($base)) {
   try {
    $attrs=[System.IO.File]::GetAttributes($p)
    if(($attrs -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {continue}
    if(($attrs -band [System.IO.FileAttributes]::Directory) -ne 0) {$stack.Push($p);continue}
    $f=[System.IO.FileInfo]::new($p);$n=$f.Length;$total+=$n;$count++
    $r=@{path=(LinuxPath $p);bytes=$n;mtime=([DateTimeOffset]$f.LastWriteTimeUtc).ToUnixTimeMilliseconds()/1000.0;owner=(Owner $p)}
    if($n -ge 262144000){$large.Add($r)}
    if($f.Extension -in @('.pt','.pth','.ckpt','.safetensors')){$checkpoints.Add($r)}
    if($f.Name -match 'sqlite|-wal|-shm|\.tmp' -or $f.Extension -eq '.log'){$special.Add($r)}
   } catch {$errors.Add($_.Exception.Message)}
  }
 } catch {$errors.Add($_.Exception.Message)}
 $sizes[(LinuxPath $base)]=@{bytes=$total;files=$count;direct_files=$count;owner=(Owner $base)}
}
foreach($p in @($sizes.Keys | Sort-Object Length -Descending)) {
 $parent=$p.Substring(0,$p.LastIndexOf('/'))
 if($sizes.ContainsKey($parent)){$sizes[$parent].bytes+=$sizes[$p].bytes;$sizes[$parent].files+=$sizes[$p].files}
}
$drive=[System.IO.DriveInfo]::new('D')
$data=@{start=$start;end=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0;capacity=$drive.TotalSize;free=$drive.AvailableFreeSpace;directories=$sizes;large_files=$large;special=$special;checkpoints=$checkpoints;errors=$errors;method='Native NTFS enumeration, excludes reparse points and audit directory; apparent sizes'}
[System.IO.File]::WriteAllText("$out\native_census.json",($data | ConvertTo-Json -Depth 10 -Compress))
Write-Output ($data | Select-Object start,end,capacity,free,errors | ConvertTo-Json -Depth 4)
