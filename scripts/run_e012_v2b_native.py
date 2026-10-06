"""Run the unchanged scientific stages with native NTFS I/O and Linux MMseqs.

No package installation, training, or change to selection/protected policies.
Recorded paths remain Linux paths so checkpoints can move between runtimes.
"""
import argparse
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types

DISTRO='Ubuntu'
LINUX_REPO='/home/simostocco/proteinGen-causal-rope'
LINUX_ROOT='/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2'

def canonical(value):
    if not isinstance(value,str):
        return value
    text=value.replace('\\','/')
    if re.match(r'^[A-Za-z]:/',text):
        return '/mnt/'+text[0].lower()+text[2:]
    for prefix in [f'//wsl.localhost/{DISTRO}',f'//wsl$/{DISTRO}']:
        if text.lower().startswith(prefix.lower()+'/'):
            return text[len(prefix):]
    return value

def native(value):
    if os.name!='nt' or not isinstance(value,str):
        return value
    if re.match(r'^/mnt/[a-z]/',value):
        return str(Path(value[5].upper()+':/'+value[7:]))
    if value.startswith(('/home/','/tmp/')):
        return str(Path('//wsl.localhost/'+DISTRO+value))
    return value

def tree(value,convert):
    if isinstance(value,dict):
        return {convert(k):tree(v,convert) for k,v in value.items()}
    if isinstance(value,list):
        return [tree(v,convert) for v in value]
    if isinstance(value,tuple):
        return tuple(tree(v,convert) for v in value)
    return convert(value)

class PortableJSON:
    def loads(self,value,*args,**kwargs):
        return tree(json.loads(value,*args,**kwargs),native)
    def dumps(self,value,*args,**kwargs):
        return json.dumps(tree(value,canonical),*args,**kwargs)

def install(module):
    module.json=PortableJSON()

def load():
    if os.name=='nt':
        import psutil
        shim=types.ModuleType('resource')
        shim.RUSAGE_SELF=0
        shim.RUSAGE_CHILDREN=-1
        def usage(who):
            peak=psutil.Process().memory_info().peak_wset//1024 if who==0 else 0
            return types.SimpleNamespace(ru_maxrss=peak)
        shim.getrusage=usage
        sys.modules['resource']=shim
        original=subprocess.Popen
        class BridgedPopen(original):
            def __init__(self,args,*positional,**kwargs):
                if isinstance(args,(list,tuple)) and str(args[0]) in {'git','mmseqs','/usr/bin/time'}:
                    cwd=canonical(str(kwargs.pop('cwd',LINUX_REPO)))
                    args=['C:/Windows/System32/wsl.exe','--distribution',DISTRO,'--cd',cwd,'--exec']+[canonical(str(a)) for a in args]
                super().__init__(args,*positional,**kwargs)
        subprocess.Popen=BridgedPopen
        # Linux MMseqs creates POSIX symlinks that Windows rmtree cannot traverse.
        import shutil
        def remove_tree(path,ignore_errors=False,onerror=None,*,onexc=None,dir_fd=None):
            if dir_fd is not None:
                raise ValueError('Native cleanup does not support relative directory descriptors')
            command=['C:/Windows/System32/wsl.exe','--distribution',DISTRO,'--exec',
                '/home/simostocco/miniforge3/envs/proteingen/bin/python','-B','-c',
                'import shutil,sys; shutil.rmtree(sys.argv[1],ignore_errors=sys.argv[2]=="1")',
                canonical(str(path)),'1' if ignore_errors else '0']
            subprocess.check_call(command)
        shutil.rmtree=remove_tree
    path=Path(native(LINUX_REPO))/'scripts/prepare_e012_sequence_corpus_v2b.py'
    spec=importlib.util.spec_from_file_location('e012_v2b_scientific_core',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install(module)
    return module

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['reservoir','screen','verify-protected','final','certify','diversity','handoff','all','tests'])
    args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['OMP_NUM_THREADS']='8'
    os.environ['MKL_NUM_THREADS']='8'
    m=load()
    if args.stage=='tests':
        import unittest
        path=Path(native(LINUX_REPO))/'tests/test_e012_data_v2b.py'
        spec=importlib.util.spec_from_file_location('native_integrity_tests',path)
        t=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(t)
        install(t.m)
        t.json=t.m.json
        result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(t))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    b=m.Build(Path(native(LINUX_ROOT)),Path(native(LINUX_REPO)))
    log=(b.root/'stats/native_build.log').open('a',buffering=1)
    class Tee:
        def __init__(self,stream):
            self.stream=stream
        def write(self,value):
            log.write(value)
            return self.stream.write(value)
        def flush(self):
            log.flush()
            self.stream.flush()
        @property
        def encoding(self):
            return self.stream.encoding
    sys.stdout=Tee(sys.stdout)
    sys.stderr=Tee(sys.stderr)
    environment={'runtime':'native Windows Python with Linux MMseqs bridge',
        'python':sys.version,'sqlite':m.sqlite3.sqlite_version,
        'packages':{p:importlib.metadata.version(p) for p in ['pyarrow','torch','numpy','psutil']},
        'cpu_threads':8,'available_ram_bytes':__import__('psutil').virtual_memory().available,
        'parent_peak_ram_measurement':'Windows PeakWorkingSetSize in KiB',
        'external_peak_ram_measurement':'Linux GNU time -v per command',
        'children_lifetime_maxrss_not_available_on_windows':True,
        'native_adapter_sha256':m.digest(__file__),
        'processing_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.repo,text=True).strip(),
        'training_launched':False}
    m.save(b.root/'stats/native_environment.json',environment)
    m.save(b.report/'native_environment.json',environment)
    stages={'reservoir':b.filter_reservoir,'screen':b.screen,
        'verify-protected':lambda:b.screen(True),'final':b.final,
        'certify':b.certify,'diversity':b.diversity,'handoff':b.handoff}
    status=b.root/'stats/build_status.json'
    m.save(status,{'status':'BUILD_RUNNING','runtime':'native Windows','native_pid':os.getpid(),
        'processing_commit':environment['processing_commit'],'timestamp':m.now(),'training_launched':False,'classification':None})
    try:
        b.verify_raw()
        if args.stage=='all':
            for name,stage in stages.items():
                m.save(status,{'status':'BUILD_RUNNING','stage':name,'runtime':'native Windows',
                    'native_pid':os.getpid(),'timestamp':m.now(),'training_launched':False,'classification':None})
                if name=='reservoir' and b.reservoir.exists() and not b.marker('reservoir').exists():
                    with b.stage('sequential_checkpoint_cache_warm'):
                        with b.reservoir.open('rb') as f:
                            while f.read(8*1024*1024):
                                pass
                if name in {'final','certify'}:
                    assignment=b.root/'tmp/shard_assignments.sqlite'
                    if assignment.exists():
                        with b.stage('sequential_assignment_cache_warm'):
                            with assignment.open('rb') as f:
                                while f.read(8*1024*1024):
                                    pass
                stage()
        else:
            stages[args.stage]()
    except BaseException as exc:
        match=re.search(r'DATA2-[CDEF]',str(exc))
        m.save(status,{'status':'STAGE_FAILED','stage':args.stage,'reason':str(exc),
            'classification':match.group() if match else None,'timestamp':m.now(),'training_launched':False})
        raise
    if args.stage in {'all','handoff'}:
        handoff=m.json.loads((b.report/'chatgpt_handoff.json').read_text())
        m.save(status,{'status':'FROZEN_CORPUS_READY','classification':handoff['classification'],
            'n':handoff['final_corpus_n'],'timestamp':m.now(),'training_launched':False})

if __name__=='__main__':
    main()
