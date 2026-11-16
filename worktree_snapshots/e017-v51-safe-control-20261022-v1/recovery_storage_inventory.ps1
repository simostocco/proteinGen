$ErrorActionPreference='Stop'
$out='D:\Simone\proteinGen_data\e017_v51_safe_control_20261022_v1\recovery_storage_20261022'
[IO.Directory]::CreateDirectory($out) | Out-Null
$env:TEMP='D:\Simone\proteinGen_data\e017_v51_safe_control_20261022_v1\runtime\tmp'
$env:TMP=$env:TEMP
[Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.Win32.SafeHandles;
public class PgDiskInventory {
 [StructLayout(LayoutKind.Sequential)] public struct Info {
  public uint Attributes; public System.Runtime.InteropServices.ComTypes.FILETIME Creation,Access,Write;
  public uint Volume,SizeHigh,SizeLow,Links,IndexHigh,IndexLow;
 }
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern SafeFileHandle CreateFileW(string p,uint access,uint share,IntPtr sa,uint disposition,uint flags,IntPtr template);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool GetFileInformationByHandle(SafeFileHandle h,out Info i);
 [StructLayout(LayoutKind.Sequential)] public struct Standard {public long Allocation,End;public uint Links;public byte DeletePending,Directory;}
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool GetFileInformationByHandleEx(SafeFileHandle h,int cls,out Standard i,uint size);
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern uint GetCompressedFileSizeW(string p,out uint high);
 public class Row { public string path; public long logical,allocated,unique_allocated,files,links,reparse; public long newest_unix; }
 public class Large { public string path,id; public long logical,allocated,mtime_unix; public uint nlink; }
 public class Link { public string id; public long allocated,logical; public uint nlink; public List<string> paths=new List<string>(); }
 public Dictionary<string,Row> directories=new Dictionary<string,Row>(StringComparer.OrdinalIgnoreCase);
 public List<Large> large_files=new List<Large>();
 public List<Large> cache_files=new List<Large>();
 public Dictionary<string,Link> hardlinks=new Dictionary<string,Link>();
 public List<string> errors=new List<string>();
 public List<string> exclusions=new List<string>();
 public long files,unique_allocated,logical,allocated;
 static string L(string p) { return "/mnt/d"+p.Substring(2).Replace('\\','/'); }
 public void Scan(string root) {
  var stack=new Stack<string>();stack.Push(root);
  var seen=new HashSet<string>();
  while(stack.Count>0) {
   string dir=stack.Pop();var row=new Row();row.path=L(dir);directories[row.path]=row;
   try {
    foreach(var f in new DirectoryInfo(dir).EnumerateFileSystemInfos()) {
     try {
      if((f.Attributes&FileAttributes.ReparsePoint)!=0) {row.reparse++;continue;}
      if((f.Attributes&FileAttributes.Directory)!=0) {
       if(f.Name.Equals("TEST",StringComparison.OrdinalIgnoreCase) || f.Name.Equals("$RECYCLE.BIN",StringComparison.OrdinalIgnoreCase) || f.Name.Equals("System Volume Information",StringComparison.OrdinalIgnoreCase)) {exclusions.Add(L(f.FullName));continue;}
       stack.Push(f.FullName);continue;
      }
      var file=(FileInfo)f;long size=file.Length;uint high;uint low=GetCompressedFileSizeW(file.FullName,out high);
      long alloc=(low==UInt32.MaxValue&&Marshal.GetLastWin32Error()!=0)?-1:((long)high<<32)|low;
      Info i;string id="UNKNOWN:"+file.FullName;uint nlink=1;
      using(var h=CreateFileW(file.FullName,0,7,IntPtr.Zero,3,0,IntPtr.Zero)) {
       if(!h.IsInvalid&&GetFileInformationByHandle(h,out i)) {
        id=i.Volume.ToString()+":"+i.IndexHigh.ToString()+":"+i.IndexLow.ToString();nlink=i.Links;
        Standard si;
        if((f.Attributes&(FileAttributes.Compressed|FileAttributes.SparseFile))==0) {
         if(GetFileInformationByHandleEx(h,1,out si,(uint)Marshal.SizeOf(typeof(Standard)))) alloc=si.Allocation;
         else {errors.Add("standard allocation unknown: "+L(file.FullName));alloc=-1;}
        }
       }
       else errors.Add("file identity: "+L(file.FullName));
      }
      if(alloc<0) {errors.Add("allocation unknown: "+L(file.FullName));alloc=0;}
      long unique=seen.Add(id)?alloc:0;
      // One shared seen-set per entire D: scan, so cross-directory hardlinks are counted once.
      row.logical+=size;row.allocated+=alloc;row.unique_allocated+=unique;row.files++;files++;logical+=size;allocated+=alloc;unique_allocated+=unique;
      long mtime=new DateTimeOffset(file.LastWriteTimeUtc).ToUnixTimeSeconds();row.newest_unix=Math.Max(row.newest_unix,mtime);
      string path=L(file.FullName);
      if(nlink>1) {Link link;if(!hardlinks.TryGetValue(id,out link)) {link=new Link();link.id=id;link.allocated=alloc;link.logical=size;link.nlink=nlink;hardlinks[id]=link;}link.paths.Add(path);row.links++;}
      var item=new Large();item.path=path;item.id=id;item.logical=size;item.allocated=alloc;item.nlink=nlink;item.mtime_unix=mtime;
      if(size>=64L*1024*1024) large_files.Add(item);
      if((path.Contains("/runtime/")||path.Contains("/tmp/")||path.Contains("/cache/")||path.Contains("/__pycache__/")||path.Contains("/.pytest_cache/"))&&size>=1024*1024) cache_files.Add(item);
      if(files%10000==0) {Console.WriteLine("INVENTORY "+files+" files");Thread.Sleep(10);}
     } catch(Exception ex) {errors.Add(L(f.FullName)+": "+ex.Message);}
    }
   } catch(Exception ex) {errors.Add(row.path+": "+ex.Message);}
  }
  var paths=new List<string>(directories.Keys);paths.Sort((a,b)=>b.Length.CompareTo(a.Length));
  foreach(string p in paths) {int cut=p.LastIndexOf('/');string parent=p.Substring(0,cut);Row pr;if(directories.TryGetValue(parent,out pr)) {Row r=directories[p];pr.logical+=r.logical;pr.allocated+=r.allocated;pr.unique_allocated+=r.unique_allocated;pr.files+=r.files;pr.links+=r.links;pr.reparse+=r.reparse;pr.newest_unix=Math.Max(pr.newest_unix,r.newest_unix);}}
 }
}
'@
$start=[DateTime]::UtcNow.ToString('o')
$drive=[IO.DriveInfo]::new('D')
$before=$drive.AvailableFreeSpace
$scan=[PgDiskInventory]::new()
$scan.Scan('D:\')
$data=@{started_utc=$start;finished_utc=[DateTime]::UtcNow.ToString('o');free_before_bytes=$before;free_after_bytes=$drive.AvailableFreeSpace;total_bytes=$drive.TotalSize;method='One below-normal single-threaded D: metadata walk. FileStandardInfo.AllocationSize for ordinary files; GetCompressedFileSizeW for compressed/sparse files; NTFS volume/file IDs deduplicate hardlinks globally. No reparse traversal, TEST directories or payload reads. Metadata errors and excluded areas remain unaccounted.';files=$scan.files;logical=$scan.logical;allocated_paths=$scan.allocated;unique_allocated=$scan.unique_allocated;directories=@($scan.directories.Values);large_files=@($scan.large_files);cache_files=@($scan.cache_files);hardlinks=@($scan.hardlinks.Values);errors=@($scan.errors);exclusions=@($scan.exclusions)}
[IO.File]::WriteAllText("$out\storage_inventory.json",($data|ConvertTo-Json -Depth 7 -Compress))
Write-Output (@{files=$scan.files;free_bytes=$drive.AvailableFreeSpace;unique_allocated=$scan.unique_allocated;errors=$scan.errors.Count;excluded=$scan.exclusions.Count;finished=$data.finished_utc}|ConvertTo-Json -Compress)
