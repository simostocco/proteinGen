param([string]$Root='D:\',[string]$Name='storage_inventory.json')
$ErrorActionPreference='Stop'
$out='D:\Simone\proteinGen_data\e017_v51_safe_control_20261022_v1\recovery_storage_20261022'
$env:TEMP='D:\Simone\proteinGen_data\e017_v51_safe_control_20261022_v1\runtime\tmp';$env:TMP=$env:TEMP
[Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
Add-Type -TypeDefinition @'
using System;using System.IO;using System.Collections.Generic;using System.Runtime.InteropServices;using Microsoft.Win32.SafeHandles;
public class PgFastAllocation {
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern SafeFileHandle CreateFileW(string p,uint access,uint share,IntPtr sa,uint disposition,uint flags,IntPtr template);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool GetFileInformationByHandleEx(SafeFileHandle h,int cls,IntPtr b,uint size);
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern uint GetCompressedFileSizeW(string p,out uint high);
 [StructLayout(LayoutKind.Sequential)] public struct Info {public uint A;public System.Runtime.InteropServices.ComTypes.FILETIME C,T,W;public uint Volume,Hi,Lo,Links,IndexHi,IndexLo;}
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool GetFileInformationByHandle(SafeFileHandle h,out Info i);
 public class Row {public string path;public long logical,allocated,unique_allocated,files,reparse,newest_unix;}
 public class File {public string path,id;public long logical,allocated,mtime_unix;public uint nlink;}
 public class Link {public string id;public long logical,allocated;public uint nlink;public List<string> paths=new List<string>();}
 public Dictionary<string,Row> directories=new Dictionary<string,Row>(StringComparer.OrdinalIgnoreCase);
 public List<File> large_files=new List<File>(),cache_files=new List<File>();public Dictionary<string,Link> hardlinks=new Dictionary<string,Link>();
 public List<string> errors=new List<string>(),exclusions=new List<string>();public long files,logical,allocated,unique_allocated;
 Dictionary<string,File> seen=new Dictionary<string,File>();
 static string L(string p){return "/mnt/d"+p.Substring(2).Replace('\\','/');}
 public void Scan(string root) {
  var stack=new Stack<string>();stack.Push(root);IntPtr buffer=Marshal.AllocHGlobal(65536);
  try {
   while(stack.Count>0) {
    string dir=stack.Pop();var row=new Row();row.path=L(dir);directories[row.path]=row;
    try {
     using(var handle=CreateFileW(dir,1,7,IntPtr.Zero,3,0x02000000,IntPtr.Zero)) {
      if(handle.IsInvalid){errors.Add(row.path+": directory open "+Marshal.GetLastWin32Error());continue;}
      bool first=true;
      while(true) {
       if(!GetFileInformationByHandleEx(handle,first?20:19,buffer,65536)) {int error=Marshal.GetLastWin32Error();if(error!=18)errors.Add(row.path+": enumerate "+error);break;}
       first=false;int pos=0;
       while(true) {
        IntPtr b=IntPtr.Add(buffer,pos);int next=Marshal.ReadInt32(b,0);uint attr=(uint)Marshal.ReadInt32(b,56);int length=Marshal.ReadInt32(b,60);
        if(length<0||length>32768)throw new Exception("invalid directory filename length");
        string name=Marshal.PtrToStringUni(IntPtr.Add(b,88),length/2);
        if(name!="."&&name!="..") {
         string full=Path.Combine(dir,name),path=L(full);
         if((attr&0x400)!=0)row.reparse++;
         else if((attr&0x10)!=0) {
          if(name.Equals("TEST",StringComparison.OrdinalIgnoreCase)||name.Equals("$RECYCLE.BIN",StringComparison.OrdinalIgnoreCase)||name.Equals("System Volume Information",StringComparison.OrdinalIgnoreCase))exclusions.Add(path);
          else stack.Push(full);
         } else {
          var item=new File();item.path=path;item.logical=Marshal.ReadInt64(b,40);item.allocated=Marshal.ReadInt64(b,48);
          byte[] id=new byte[16];Marshal.Copy(IntPtr.Add(b,72),id,0,16);item.id=BitConverter.ToString(id);
          item.mtime_unix=(Marshal.ReadInt64(b,24)-116444736000000000L)/10000000L;
          if((attr&(0x800|0x200))!=0 && item.allocated>0) {
           uint high;uint low=GetCompressedFileSizeW(full,out high);
           if(low==UInt32.MaxValue&&Marshal.GetLastWin32Error()!=0)errors.Add(path+": compressed allocation unknown");
           else item.allocated=((long)high<<32)|low;
          }
          if(item.logical>=64L*1024*1024) {
           using(var h=CreateFileW(full,0,7,IntPtr.Zero,3,0,IntPtr.Zero)){Info inf;if(!h.IsInvalid&&GetFileInformationByHandle(h,out inf))item.nlink=inf.Links;}
           large_files.Add(item);
          }
          File previous;long unique=item.allocated;
          if(seen.TryGetValue(item.id,out previous)) {
           unique=0;Link link;if(!hardlinks.TryGetValue(item.id,out link)) {link=new Link();link.id=item.id;link.logical=item.logical;link.allocated=item.allocated;link.nlink=Math.Max(previous.nlink,item.nlink);link.paths.Add(previous.path);hardlinks[item.id]=link;}link.paths.Add(path);
          } else {seen[item.id]=item;if(item.nlink>1){var link=new Link();link.id=item.id;link.logical=item.logical;link.allocated=item.allocated;link.nlink=item.nlink;link.paths.Add(path);hardlinks[item.id]=link;}}
          row.logical+=item.logical;row.allocated+=item.allocated;row.unique_allocated+=unique;row.files++;row.newest_unix=Math.Max(row.newest_unix,item.mtime_unix);
          files++;logical+=item.logical;allocated+=item.allocated;unique_allocated+=unique;
          if((path.Contains("/runtime/")||path.Contains("/tmp/")||path.Contains("/cache/")||path.Contains("/__pycache__/"))&&item.logical>=1024*1024)cache_files.Add(item);
          if(files%100000==0)Console.WriteLine("NATIVE_BATCHED "+files+" files");
         }
        }
        if(next==0)break;pos+=next;if(pos>=65536)throw new Exception("directory offset outside buffer");
       }
      }
     }
    }catch(Exception ex){errors.Add(row.path+": "+ex.Message);}
   }
  }finally{Marshal.FreeHGlobal(buffer);}
  var paths=new List<string>(directories.Keys);paths.Sort((a,b)=>b.Length.CompareTo(a.Length));
  foreach(string p in paths){string parent=p.Substring(0,p.LastIndexOf('/'));Row pr;if(directories.TryGetValue(parent,out pr)){Row r=directories[p];pr.logical+=r.logical;pr.allocated+=r.allocated;pr.unique_allocated+=r.unique_allocated;pr.files+=r.files;pr.reparse+=r.reparse;pr.newest_unix=Math.Max(pr.newest_unix,r.newest_unix);}}
 }
}
'@
$drive=[IO.DriveInfo]::new('D');$before=$drive.AvailableFreeSpace;$start=[DateTime]::UtcNow.ToString('o')
$scan=[PgFastAllocation]::new();$scan.Scan($Root)
$data=@{started_utc=$start;finished_utc=[DateTime]::UtcNow.ToString('o');free_before_bytes=$before;free_after_bytes=$drive.AvailableFreeSpace;total_bytes=$drive.TotalSize;method='Single-threaded below-normal batched FileIdExtdDirectoryInfo NTFS enumeration: native allocation sizes and 128-bit file IDs; global hardlink deduplication. GetCompressedFileSizeW corrects sparse/compressed allocation. Link counts queried for >=64MiB files. No payload reads, reparse traversal or TEST-directory entry. Excluded areas/errors remain unaccounted.';files=$scan.files;logical=$scan.logical;allocated_paths=$scan.allocated;unique_allocated=$scan.unique_allocated;directories=@($scan.directories.Values);large_files=@($scan.large_files);cache_files=@($scan.cache_files);hardlinks=@($scan.hardlinks.Values);errors=@($scan.errors);exclusions=@($scan.exclusions)}
[IO.File]::WriteAllText("$out\$Name",($data|ConvertTo-Json -Depth 7 -Compress))
Write-Output (@{files=$scan.files;free_bytes=$drive.AvailableFreeSpace;allocated=$scan.unique_allocated;errors=$scan.errors.Count;finished=$data.finished_utc}|ConvertTo-Json -Compress)
