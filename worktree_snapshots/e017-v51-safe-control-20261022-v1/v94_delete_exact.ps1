$ErrorActionPreference='Stop'
$C='D:\Simone\proteinGen_data\e017_v94_fixed_sequence_geometry_rl_20261024_v1\uniref_cleanup_20261024_v1'
$env:TEMP=$C+'\runtime_windows_tmp';$env:TMP=$env:TEMP
[Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
Add-Type -TypeDefinition @'
using System;using System.IO;using System.Runtime.InteropServices;using System.Text;using System.Collections.Generic;using Microsoft.Win32.SafeHandles;
public class ExactUniRef {
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)]static extern SafeFileHandle CreateFileW(string p,uint a,uint s,IntPtr x,uint d,uint f,IntPtr t);
 [StructLayout(LayoutKind.Sequential)]public struct Info {public uint Attr;public System.Runtime.InteropServices.ComTypes.FILETIME C,A,W;public uint Volume,Hi,Lo,Links,IndexHi,IndexLo;}
 [DllImport("kernel32.dll",SetLastError=true)]static extern bool GetFileInformationByHandle(SafeFileHandle h,out Info i);
 [DllImport("kernel32.dll",SetLastError=true)]static extern bool GetFileInformationByHandleEx(SafeFileHandle h,int c,IntPtr b,uint n);
 [StructLayout(LayoutKind.Sequential)]struct Disposition {[MarshalAs(UnmanagedType.Bool)]public bool Delete;}
 [DllImport("kernel32.dll",SetLastError=true)]static extern bool SetFileInformationByHandle(SafeFileHandle h,int c,ref Disposition b,uint n);
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)]static extern bool GetDiskFreeSpaceExW(string p,out ulong a,out ulong t,out ulong f);
 [StructLayout(LayoutKind.Sequential)]public struct Unique {public int PID;public System.Runtime.InteropServices.ComTypes.FILETIME Start;}
 [StructLayout(LayoutKind.Sequential,CharSet=CharSet.Unicode)]public struct Process {public Unique P;[MarshalAs(UnmanagedType.ByValTStr,SizeConst=256)]public string App;[MarshalAs(UnmanagedType.ByValTStr,SizeConst=64)]public string Service;public int Type;public uint Status,Session;[MarshalAs(UnmanagedType.Bool)]public bool Restartable;}
 [DllImport("rstrtmgr.dll",CharSet=CharSet.Unicode)]static extern int RmStartSession(out uint s,uint flags,StringBuilder key);
 [DllImport("rstrtmgr.dll",CharSet=CharSet.Unicode)]static extern int RmRegisterResources(uint s,uint n,string[] paths,uint apps,Unique[] p,uint svcs,string[] svc);
 [DllImport("rstrtmgr.dll")]static extern int RmGetList(uint s,out uint need,ref uint n,[In,Out]Process[] p,ref uint reason);
 [DllImport("rstrtmgr.dll")]static extern int RmEndSession(uint s);
 public class Meta {public string FileId64;public uint Volume,Links,Attributes;public long Logical,Allocated,WriteFileTime;}
 public static FileStream Hold(string p) {var h=CreateFileW(p,0x80010000,1,IntPtr.Zero,3,0x80,IntPtr.Zero);if(h.IsInvalid)throw new Exception("exclusive read/delete open failed "+Marshal.GetLastWin32Error());return new FileStream(h,FileAccess.Read,1048576,false);}
 public static Meta Query(SafeFileHandle h) {Info i;if(!GetFileInformationByHandle(h,out i))throw new Exception("identity query failed");IntPtr b=Marshal.AllocHGlobal(24);try{if(!GetFileInformationByHandleEx(h,1,b,24))throw new Exception("allocation query failed");return new Meta{FileId64=(((ulong)i.IndexHi<<32)|i.IndexLo).ToString(),Volume=i.Volume,Links=i.Links,Attributes=i.Attr,Logical=Marshal.ReadInt64(b,8),Allocated=Marshal.ReadInt64(b,0),WriteFileTime=((long)i.W.dwHighDateTime<<32)|(uint)i.W.dwLowDateTime};}finally{Marshal.FreeHGlobal(b);}}
 public static ulong Free() {ulong a,t,f;if(!GetDiskFreeSpaceExW("D:\\",out a,out t,out f))throw new Exception("capacity query failed");return a;}
 public static Process[] Users(string path) {uint s;int err=RmStartSession(out s,0,new StringBuilder(33));if(err!=0)throw new Exception("RmStart "+err);try{err=RmRegisterResources(s,1,new string[]{path},0,null,0,null);if(err!=0)throw new Exception("RmRegister "+err);uint need,n=0,reason=0;err=RmGetList(s,out need,ref n,null,ref reason);if(err==0)return new Process[0];if(err!=234)throw new Exception("RmGetList "+err);var p=new Process[need];n=need;err=RmGetList(s,out need,ref n,p,ref reason);if(err!=0)throw new Exception("RmGetList second "+err);return p;}finally{RmEndSession(s);}}
 public static void DeleteSameHandle(SafeFileHandle h) {var d=new Disposition{Delete=true};if(!SetFileInformationByHandle(h,4,ref d,4))throw new Exception("delete disposition failed "+Marshal.GetLastWin32Error());}
}
'@
function Save([string]$Name,$Value) {$p=Join-Path $C $Name;$tmp=$p+'.part';if(Test-Path -LiteralPath $tmp){throw 'staging collision'};$bytes=[Text.Encoding]::UTF8.GetBytes(($Value|ConvertTo-Json -Depth 12 -Compress));$s=[IO.FileStream]::new($tmp,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);try{$s.Write($bytes,0,$bytes.Length);$s.Flush($true)}finally{$s.Dispose()};[IO.File]::Move($tmp,$p)}
$allow=Get-Content -LiteralPath ($C+'\DELETE_ALLOWLIST.json') -Raw|ConvertFrom-Json
$proof=Get-Content -LiteralPath ($C+'\PROTECTED_AND_DEPENDENCY_VERIFICATION.json') -Raw|ConvertFrom-Json
if(!$proof.PASS -or $allow.entries.Count -ne 1){throw 'preflight not passed/exact allowlist length mismatch'}
$item=$allow.entries[0];if($item.classification -ne 'DISPOSABLE_UNIREF_DATASET'){throw 'unauthorized classification'}
$path=[IO.Path]::GetFullPath($item.native_path)
$expected='D:\Simone\proteinGen_data\sequence_foundation\uniref50_2026_03_e012_v2\v2c_optimization\exports\candidate_payload.bin'
if($path -cne $expected){throw 'exact filename/boundary mismatch'}
$dir=[IO.DirectoryInfo]::new([IO.Path]::GetDirectoryName($path));while($dir){if(($dir.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw 'reparse ancestor'};$dir=$dir.Parent}
if(([IO.File]::GetAttributes($path) -band ([IO.FileAttributes]::ReparsePoint -bor [IO.FileAttributes]::Directory)) -ne 0){throw 'not regular nonsymlink payload'}
$foreign=@([ExactUniRef]::Users($path)|Where-Object {$_.P.PID -ne $PID});if($foreign.Count){throw ('active Windows users: '+($foreign|ConvertTo-Json -Depth 5 -Compress))}
$fs=[ExactUniRef]::Hold($path)
try {
 $identity=[ExactUniRef]::Query($fs.SafeFileHandle)
 if($identity.Logical -ne $item.apparent_bytes -or $identity.Allocated -ne $item.allocated_bytes -or $identity.Links -ne 1 -or $identity.FileId64 -ne [string]$item.native_file_id64){throw 'identity/allocation/hardlink mismatch'}
 $hasher=[Security.Cryptography.SHA256]::Create();try{$actual=([BitConverter]::ToString($hasher.ComputeHash($fs))).Replace('-','').ToLowerInvariant()}finally{$hasher.Dispose()}
 if($actual -ne $item.expected_SHA256){throw 'payload SHA256 mismatch'}
 $again=[ExactUniRef]::Query($fs.SafeFileHandle);if(($again|ConvertTo-Json -Compress) -ne ($identity|ConvertTo-Json -Compress)){throw 'file changed during hash'}
 $foreign=@([ExactUniRef]::Users($path)|Where-Object {$_.P.PID -ne $PID});if($foreign.Count){throw 'Windows handle dependency changed'}
 $binding=@{PASS=$true;path=$item.path;SHA256=$actual;identity=$identity;classification=$item.classification;native_users_excluding_own=@();exclusive_handle='GENERIC_READ+DELETE, shareREAD only; no writers or deleters admitted';D_free_bytes=[ExactUniRef]::Free();allowlist_SHA256=(Get-FileHash -LiteralPath ($C+'\DELETE_ALLOWLIST.json') -Algorithm SHA256).Hash.ToLowerInvariant();dry_run_only=$true;deleted=$false}
 Save 'DRY_RUN.json' $binding
 Write-Output 'DRY_RUN_READY_EXCLUSIVE_HANDLE_HELD'
 $deadline=[DateTime]::UtcNow.AddMinutes(3)
 while(!(Test-Path -LiteralPath ($C+'\EXECUTE_EXACT_ALLOWLIST.json'))) {if([DateTime]::UtcNow -gt $deadline){throw 'execution receipt not supplied; no deletion'};Start-Sleep -Seconds 2}
 $execute=Get-Content -LiteralPath ($C+'\EXECUTE_EXACT_ALLOWLIST.json') -Raw|ConvertFrom-Json
 if($execute.allowlist_SHA256 -ne $binding.allowlist_SHA256 -or $execute.dry_run_SHA256 -ne (Get-FileHash -LiteralPath ($C+'\DRY_RUN.json') -Algorithm SHA256).Hash.ToLowerInvariant()){throw 'execution receipt mismatch'}
 $current=[ExactUniRef]::Query($fs.SafeFileHandle);if(($current|ConvertTo-Json -Compress) -ne ($identity|ConvertTo-Json -Compress)){throw 'identity drift before delete'}
 $foreign=@([ExactUniRef]::Users($path)|Where-Object {$_.P.PID -ne $PID});if($foreign.Count){throw 'active Windows users before delete'}
 Save 'TOMBSTONE_INTENT.json' @{path=$item.path;SHA256=$actual;identity=$identity;allowlist_SHA256=$binding.allowlist_SHA256;dry_run_SHA256=$execute.dry_run_SHA256;reason='verified lossless obsolete UniRef dataset cache; canonical reservoir and all scientific evidence retained';authorization='explicit current user UniRef payload/cache deletion instruction';exact_same_handle_deletion=$true}
 $before=[ExactUniRef]::Free();[ExactUniRef]::DeleteSameHandle($fs.SafeFileHandle);$fs.Dispose();$fs=$null
 if(Test-Path -LiteralPath $path){throw 'target still present after handle closure; stop'}
 $after=[ExactUniRef]::Free()
 Save 'BATCH_0001_RESULT.json' @{deleted=@($item.path);deleted_count=1;SHA256=$actual;identity=$identity;D_free_before_bytes=$before;D_free_after_bytes=$after;observed_free_gain_bytes=([long]$after-[long]$before);estimated_allocation_bytes=$identity.Allocated;all_other_paths_untouched=$true;deleted_directories=0;reserve_bytes=2147483648}
 Write-Output (@{DELETED=1;before=$before;after=$after;observed_gain=([long]$after-[long]$before)}|ConvertTo-Json -Compress)
}finally{if($fs){$fs.Dispose()}}
