"""CPU-only architecture report; never imports entry-point storage bootstrap.

Writes Markdown in docs and stages JSON in /tmp. Publishing the staged JSON to
the experiment root is separate. This file is outside production source hashes.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import json
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
import torch
import yaml
from torch import nn
import protein_distance_diffusion.models.e015_multiscale_parity_denoiser as architecture
from protein_distance_diffusion.training.e015_engine import Engine
from protein_distance_diffusion.training.e015_runtime import Runtime
from protein_distance_diffusion.training.e014_runtime import sha

torch.set_num_threads(1)
torch.manual_seed(15015)
config_path = WORK / 'configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml'
cfg = yaml.safe_load(config_path.read_text())
root = Path(cfg['output_dir'])
cal_path = root / 'calibration/frozen.json'
cal = json.loads(cal_path.read_text())
batch = json.loads((root / 'batch_autotuning.json').read_text())
precision = batch['selected']['385-500']['precision']
assert precision == 'bf16'
runtime = object.__new__(Runtime)
runtime.config_path = config_path
runtime.cfg = cfg
contract_before = runtime.contract(cal)
assert contract_before == json.loads((root / 'preflight/full_certification.json').read_text())['calibration_contract']
before_state = json.loads((root / 'STATE.json').read_text())
before_owner = json.loads((root / 'SUPERVISOR.json').read_text())
engine = Engine(cfg, cal, device='cpu', precision=precision)
model = engine.model
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert total == 33_128_139
assert all(p.device.type == 'cpu' for p in model.parameters())
assert not torch.cuda.is_initialized()

def metadata(x):
    if isinstance(x, torch.Tensor):
        return dict(shape=list(x.shape), dtype=str(x.dtype), device=x.device.type)
    if isinstance(x, architecture.State):
        return {key: metadata(getattr(x, key)) for key in ('h','v','p','a')}
    if isinstance(x, (tuple,list)):
        return [metadata(v) for v in x]
    if isinstance(x, dict):
        return {k:metadata(v) for k,v in x.items()}
    return x

activations = {}
observed = {}
graph_records = []
head_values = {}
handles = []
for name, module in model.named_modules():
    def hook(m, args, out, name=name):
        observed.setdefault(name, []).append(dict(inputs=metadata(args), output=metadata(out)))
        if isinstance(m, architecture.TypedBlock):
            activations[name + '.input'] = dict(zip(('h','v','p','a'), metadata(args[:4])))
            activations[name + '.output'] = dict(zip(('h','v','p','a'), metadata(out)))
        if name in ('up_medium','up_fine'):
            activations[name] = metadata(out)
        if name in ('polar_v','polar_p','chiral_a','chiral_p','local'):
            head_values[name] = out
    handles.append(module.register_forward_hook(hook))

original_prepare = model.prepare_single
original_head = model.output_head
original_graph = architecture.graph
original_pool = architecture.pool_state
pool_calls = []

def prepare(self, *args, **kwargs):
    result = original_prepare(*args, **kwargs)
    activations['fine.embedding'] = metadata(result[0])
    for key,value in zip(('frame','reliability','gate','time'), result[1:]):
        activations['fine.'+key] = metadata(value)
    return result

def pooled(state):
    result = original_pool(state)
    stage = 'fine_to_medium' if len(state.h) == 500 else 'medium_to_coarse'
    activations[stage+'.input'] = metadata(state)
    activations[stage+'.output'] = metadata(result)
    pool_calls.append(stage)
    return result

def graphed(x, previous, k, k_self, dense=False):
    result = original_graph(x, previous, k, k_self, dense)
    i,j,features,unit = result
    n=len(x)
    ids=torch.arange(n)
    separation=ids[None]-ids[:,None]
    polymer=(separation.abs()>=1)&(separation.abs()<=3)
    current=torch.zeros_like(polymer)
    sc=torch.zeros_like(polymer)
    if dense:
        current=separation!=0
    elif k:
        current.scatter_(1,architecture.stable_knn(x,k),True)
    if previous is not None and k_self:
        sc.scatter_(1,architecture.stable_knn(previous,k_self),True)
    assert int((polymer|current|sc).sum()) == len(i)
    graph_records.append(dict(stage={500:'fine',125:'medium',32:'coarse'}[n],
        coordinates=metadata(x), previous_coordinates=metadata(previous),
        polymer_edges=int(polymer.sum()), current_edges=int(current.sum()),
        self_conditioning_edges=int(sc.sum()), deduplicated_edges=len(i),
        polymer_indices_shape=[2,int(polymer.sum())],
        current_indices_shape=[2,int(current.sum())],
        self_conditioning_indices_shape=[2,int(sc.sum())],
        current_knn_indices_shape=None if dense else [n,min(k,n-1)],
        self_conditioning_knn_indices_shape=[n,min(k_self,n-1)],
        indices_i=metadata(i), indices_j=metadata(j), edge_features=metadata(features),
        unit_vectors=metadata(unit), provenance_masks_shape=[n,n],
        dense=dense, attention_heads=0 if n==500 else 1,
        query_key_shape=None if n==500 else [n,64],
        actual_attention_logits_and_weights_shape=None if n==500 else [len(i)],
        conceptual_dense_attention_shape=[n,n] if dense else None,
        dense_attention_matrix_materialized=False,
        notes='Directed edges; overlap is deduplicated, provenance is retained. Coarse attention excludes self edges.'))
    return result

def head(self, state, frame, reliability, gate, shapes):
    result=original_head(state,frame,reliability,gate,shapes)
    # Recover branch intermediates from actual projection hook outputs; no
    # additional learned layers are called.
    v,a=state.v.float(),state.a.float()
    polar=(head_values['polar_v'].float()[...,None]*v).sum(1)/(96**0.5)
    polar=polar+(head_values['polar_p'].float()[...,None]*a).sum(1)/(48**0.5)
    chiral=(head_values['chiral_a'].float()[...,None]*a).sum(1)/(48**0.5)
    chiral=chiral+(head_values['chiral_p'].float()[...,None]*v).sum(1)/(96**0.5)
    with torch.autocast('cpu',enabled=False):
        local=torch.einsum('nac,nc->na',frame.float(),head_values['local'].float())
    activations['heads.raw_branches_reconstructed_from_projection_hooks']={
        'polar':metadata(polar),'chiral':metadata(chiral),'local':metadata(local)}
    activations['heads.components']=metadata(result[1])
    activations['heads.single_v_hat']=metadata(result[0])
    return result

model.prepare_single=MethodType(prepare,model)
model.output_head=MethodType(head,model)
architecture.graph=graphed
architecture.pool_state=pooled
generator=torch.Generator().manual_seed(150150500)
x=torch.randn(1,500,3,generator=generator)
previous=torch.randn(1,500,3,generator=generator)
times=torch.tensor([50],dtype=torch.long)
lengths=torch.tensor([500],dtype=torch.long)
mask=torch.ones(1,500,dtype=torch.bool)
available=torch.ones(1,dtype=torch.bool)
activations['inputs']=metadata(dict(noisy_coordinates=x,timesteps=times,lengths=lengths,
    residue_mask=mask,x0_prev=previous,sc_available=available))
print('Factory parameter assertion passed; starting CPU B1/N500 BF16 forward',flush=True)
try:
    with torch.no_grad(), engine.autocast():
        output=model(x,times,lengths,mask,previous,available)
    assert output.shape == (1,500,3) and torch.isfinite(output).all()
    assert not output.requires_grad and all(p.grad is None for p in model.parameters())
finally:
    model.prepare_single=original_prepare
    model.output_head=original_head
    architecture.graph=original_graph
    architecture.pool_state=original_pool
    for handle in handles:
        handle.remove()
activations['final.v_hat']=metadata(output)
assert pool_calls==['fine_to_medium','medium_to_coarse']
assert len([k for k in activations if k.endswith('.output') and k.startswith(('fine.','medium.','coarse.'))])==16

def checkpoint_region(name):
    return any(name.startswith(f'{stage}.{i}') and (name==f'{stage}.{i}' or name[len(f'{stage}.{i}')]=='.')
               for stage,count in [('fine',8),('medium',4),('coarse',4)] for i in range(count))

tensor_inventory=[]
modules=[]
for name,module in model.named_modules():
    direct=list(module.named_parameters(recurse=False))
    recursive=sum(p.numel() for p in module.parameters() if p.requires_grad)
    direct_total=sum(p.numel() for _,p in direct if p.requires_grad)
    modules.append(dict(name=name or '<root>',type=type(module).__name__,direct_parameters=direct_total,
        recursive_parameters=recursive,percentage_of_total=100*recursive/total,
        direct_tensor_names=[f'{name}.{key}'.lstrip('.') for key,_ in direct],
        production_parameter_storage_dtype='torch.float32',
        production_compute=('BF16 autocast Linear GEMM output; FP32 stored weight/bias' if isinstance(module,nn.Linear) else
                            'FP32 CUDA autocast LayerNorm' if isinstance(module,nn.LayerNorm) else
                            'Mixed: BF16 learned Linear projections; FP32 geometry/reductions and CUDA LayerNorm; details in child modules and CPU observations'),
        checkpoint_region=checkpoint_region(name),checkpointed_in_this_no_grad_forward=False,
        observed_cpu_autocast_calls=observed.get(name,[])))
    for key,p in direct:
        tensor_inventory.append(dict(name=f'{name}.{key}'.lstrip('.'),module=name,shape=list(p.shape),
            parameters=p.numel(),requires_grad=p.requires_grad,storage_dtype=str(p.dtype),
            production_operation_dtype='BF16 GEMM output' if isinstance(module,nn.Linear) else
            'FP32 CUDA autocast LayerNorm' if isinstance(module,nn.LayerNorm) else 'FP32',
            checkpoint_region=checkpoint_region(name)))
assert sum(row['parameters'] for row in tensor_inventory)==total
assert sum(row['direct_parameters'] for row in modules)==total
assert len({row['name'] for row in tensor_inventory})==len(tensor_inventory)

groups=[]
def group(label,names):
    tensors=[row for row in tensor_inventory if any(row['module']==n or row['module'].startswith(n+'.') for n in names)]
    count=sum(row['parameters'] for row in tensors)
    groups.append(dict(name=label,module_paths=names,direct_parameters=0,
        recursive_parameters=count,percentage_of_total=100*count/total,
        tensor_names=[row['name'] for row in tensors],checkpoint_region=any(checkpoint_region(n) for n in names)))
group('Input/time/length/self-conditioning embeddings',['time','input','vinit','ainit','pinit'])
group('Fine edge embeddings (functional RBF/geometry; no learned edge-only module)',[])
for i in range(8):group(f'Fine block {i+1}',[f'fine.{i}'])
group('Fine-to-medium pooling (functional mean stride 4)',[])
for i in range(4):group(f'Medium block {i+1}',[f'medium.{i}'])
group('Medium-to-coarse pooling (functional mean stride 4)',[])
for i in range(4):group(f'Coarse block {i+1}',[f'coarse.{i}'])
group('Coarse-to-medium decoding',['up_medium'])
group('Medium-to-fine decoding',['up_fine'])
group('Polar output head',['polar_v','polar_p'])
group('Chiral output head',['chiral_a','chiral_p'])
group('Local-frame head',['local'])
group('Chiral/reliability gates (analytic; decoder gates counted with decoders)',[])
covered={n for g in groups for n in g['tensor_names']}
remaining=[row for row in tensor_inventory if row['name'] not in covered]
assert not remaining and sum(g['recursive_parameters'] for g in groups)==total
group('Remaining trainable modules',[])
bins=[]
for index,(values,weight) in enumerate(zip(engine.sampler.bins,cal['v_weights'],strict=True)):
    assert values.tolist()==cal['snr_bins'][index]
    logsnr=engine.sampler.logsnr[values]
    bins.append(dict(bin=index,count=len(values),timestep_min=int(values.min()),timestep_max=int(values.max()),
        timesteps_descending=values.tolist(),logsnr_min=float(logsnr.min()),logsnr_max=float(logsnr.max()),v_loss_weight=weight))
after_state=json.loads((root/'STATE.json').read_text())
after_owner=json.loads((root/'SUPERVISOR.json').read_text())
assert after_owner['pid']==before_owner['pid'] and after_owner['child_pid']==before_owner['child_pid']
assert before_state['pid']==after_state['pid'] and after_state['update']>=before_state['update']
assert runtime.contract(cal)==contract_before
assert not torch.cuda.is_initialized()

production=dict(model_config=cfg['model'],rbf_channels_per_distance=16,rbf_distance_streams=2,
    total_rbf_channels=32,edge_feature_width=39,rbf_centers=[0.0,5.0],rbf_spacing=5/15,
    rbf_gamma=1/(5/15)**2,time_input_width=33,time_hidden_width=128,time_output_width=64,
    time_sinusoidal_frequencies=16,node_embedding_input_width=76,length_feature='N/500, one scalar; no separate learned length table',
    self_conditioning_features='displacement norm, availability flag, displacement polar seed, kNN geometry; no separate learned SC table',
    rope_dimensions=64,rope_pairs=32,rope_base=10000,rope_positions='per-protein pooled residue indices reset in packed batch',
    attention_heads={'fine':0,'medium':1,'coarse':1},query_key_width=64,attention_scale=8,
    knn_current=24,knn_self_conditioning=16,polymer_offsets=[-3,-2,-1,1,2,3],
    coarse_dense=True,coarse_self_edges=False,knn_distance_dtype='torch.float64',
    knn_boundary_tolerance='32 * FP32 epsilon * max(boundary squared distance,1)',
    message_mlp={'input':1063,'hidden':768,'output':720,'expansion_ratio':cfg['model']['mlp_ratio']},
    feed_mlp={'input':384,'hidden':768,'output':384,'expansion_ratio':2},
    local_mlp={'input':384,'hidden':384,'output':3,'expansion_ratio':1},
    normalization={'scalar':'LayerNorm(384), eps=1e-5, affine=True, at block norm and feed.0',
        'polar_axial':'z / sqrt(1 + sum_xyz(z^2)), independently per channel; FP32',
        'pseudoscalar':'odd tanh','pooling':'arithmetic mean of contiguous stride-4 groups, valid counts',
        'final':'FP32 mean centering over valid residues'},
    self_conditioning_probability=cfg['self_conditioning_probability'],
    rollout_probability=cfg['rollout_probability'],rollout_warmup_exposures=cfg['rollout_warmup_exposures'],
    rollout_steps=cfg['rollout_steps'],ema_decay=cfg['ema_decay'],snr_bins=bins,v_loss_weights=cal['v_weights'],
    geometry_gates={'loss_formula':'sigmoid((logSNR - midpoint)/width)',
        'midpoints':cal['midpoints'],'loss_gate_width_logsnr':cal['gate_width_logsnr'],
        'branch_formula':'reliability * sigmoid((logSNR - chirality_midpoint)/gate_width + 2*(reliability-1))',
        'branch_gate_width':model.gate_width,'branch_midpoint':float(model.parity_midpoint),
        'reliability':'sin(local angle) * valid_frame; no trainable parameters'},
    geometry_loss_coefficients=cal['coefficients'],chirality_tau=0.1,chirality_beta_sign=0.1,
    chirality_huber_delta=1.0,clash_threshold_angstrom=3,clash_min_chain_separation_exclusive=3,
    distance_loss='mean of L1/L2/L3 squared distance errors, each coefficient lambda_distance/3',
    coordinate_scale_angstrom=cfg['coordinate_scale_angstrom'],feature_flags=cfg['features'])
assert production['message_mlp']['input']==model.fine[0].message[0].in_features
assert production['message_mlp']['output']==model.fine[0].message[-1].out_features
report=dict(schema_version=1,created_unix=time.time(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
    config_path=str(config_path),config_sha256=sha(config_path),calibration_sha256=sha(cal_path),
    source_hashes=contract_before['source_hashes'],torch_version=str(torch.__version__),
    factory='supervise_e015_training.py -> run_e015_integrated.py -> Trainer -> Engine(cfg,calibration,device="cpu",precision="bf16") -> E015MultiscaleParityDenoiser(**cfg["model"])',
    model_is_fresh_cpu_architecture_replica=True,checkpoint_loaded=False,total_trainable_parameters=total,
    production_autocast_authority=dict(torch_git_version=torch.version.git_version,
        installed_header=str(Path(torch.__file__).parent/'include/ATen/autocast_mode.h'),
        installed_header_sha256=sha(Path(torch.__file__).parent/'include/ATen/autocast_mode.h'),
        layer_norm_policy='AT_FORALL_FP32 includes layer_norm and native_layer_norm',
        documentation='https://docs.pytorch.org/docs/2.14/amp.html#cuda-ops-that-can-autocast-to-float32',
        documentation_note='General framework policy reference; installed 2.13 header is the version-specific evidence.'),
    groups=groups,modules=modules,tensors=tensor_inventory,buffers=[dict(name=n,**metadata(v),elements=v.numel()) for n,v in model.named_buffers()],
    activation_trace=activations,graphs=graph_records,production_values=production,
    synthetic=dict(seed=150150500,batch=1,length=500,timestep=50,context='independent synthetic Gaussian FP32 x0_prev with availability true',
        model_seed=15015,gradient_enabled=False,cpu_autocast='bf16',finite_output=True,
        checkpointing_exercised=False,production_checkpoint_condition='model.training and activation_checkpointing and torch.is_grad_enabled()',
        dtype_scope='Parameter storage and CUDA precision policy are sourced from production code. Activation dtypes are observed on CPU BF16 autocast, not live CUDA instrumentation.'),
    noninterference=dict(cuda_initialized=False,trainer_pid=after_state['pid'],supervisor_pid=after_owner['pid'],
        trainer_before_update=before_state['update'],trainer_after_update=after_state['update'],
        production_contract_unchanged=True,trainer_signals_sent=False,checkpoint_writes=False,
        production_optimizer_accessed=False,scientific_protocol_modified=False,
        note='Only fresh CPU objects and audit report files are touched. Trainer can naturally update checkpoints/optimizer during the audit.'))

diagram='''flowchart TD
    I["X_t, t, N, masks, optional X0_prev"] --> E["Time 33→128→64; node 76→384; typed seeds"]
    E --> F["Fine: N=500, H/V/P/A=384/96/48/48; 8 blocks"]
    G["Polymer ±1/2/3 ∪ current kNN24 ∪ SC kNN16; 39 edge features"] --> F
    F --> PM["Mean pool ×4: 125 nodes"] --> M["4 medium blocks; sparse 1-head RoPE64 attention"]
    M --> PC["Mean pool ×4: 32 nodes"] --> C["4 coarse blocks; dense off-diagonal 1-head RoPE64 attention"]
    C --> UM["Gated coarse→medium Broadcast"]
    M --> UM
    UM --> UF["Gated medium→fine Broadcast"]
    F --> UF
    UF --> P["Polar head"]
    UF --> CH["Chiral head"]
    UF --> L["Local-frame head"]
    R["Analytic logSNR/reliability gate"] --> CH
    R --> L
    P --> O["FP32 centered branch sum: v_hat [1,500,3]"]
    CH --> O
    L --> O'''
plain='''Inputs [1,500,3], t/N [1], mask [1,500], SC coordinates [1,500,3]
  -> time [64] + scalar geometry [500,12] -> H/V/P/A embedding
  -> FINE [500]: 8 typed blocks (polymer + kNN24 + SC kNN16)
       | skip                                  |
       |                               mean pool stride4
       |                                       v
       |                       MEDIUM [125]: 4 RoPE64 attention blocks
       |                              | skip           |
       |                              |         mean pool stride4
       |                              |                v
       |                              |       COARSE [32]: 4 dense attention blocks
       |                              +<-- gated Broadcast --+
       +<---------------- gated Broadcast ----+
  -> polar [500,3] + gated chiral [500,3] + gated local-frame [500,3]
  -> FP32 valid-residue centering -> v_hat [1,500,3]'''
lines=['# E015 architecture and parameter audit','',
    f'Exact production architecture: **{total:,} trainable parameters**, asserted from the CPU model created by the production `Engine` factory. Audited source commit `{report["git_commit"]}`.', '',
    'The supervisor launches the trainer; it does not construct a model itself. This audit calls the same Engine constructor with the exact YAML model arguments, frozen calibrated parity midpoint, and selected BF16 policy. It constructs a fresh CPU replica, not the running learned weights. No checkpoint is loaded or rewritten. The production source/config/calibration contract matches the current certificate before and after the audit.', '',
    f'Config SHA256: `{report["config_sha256"]}`. Calibration SHA256: `{report["calibration_sha256"]}`. Torch: `{report["torch_version"]}`.', '',
    '## Architecture','', '```mermaid',diagram,'```','','```text',plain,'```','',
    '## Disjoint parameter accounting','',
    'Counts below sum exactly to the total. Group containers/functional operations have zero direct parameters; their recursive totals contain the listed module paths. Learned edge processing lives inside each block message MLP, with no separate edge embedding weights. Length and self-conditioning use shared input features/seeds, with no standalone learned tables. Decoder gates are included in their decoders. Buffers are excluded.', '',
    '| Component | Module paths | Direct | Recursive | % total | Checkpoint region |',
    '|---|---|---:|---:|---:|---|']
for row in groups:
    lines.append(f'| {row["name"]} | {", ".join(row["module_paths"]) or "functional/none"} | {row["direct_parameters"]:,} | {row["recursive_parameters"]:,} | {row["percentage_of_total"]:.8f}% | {row["checkpoint_region"]} |')
lines += ['',f'**Total: {sum(g["recursive_parameters"] for g in groups):,}.**','',
    '## Precision and checkpoint semantics','',
    'All trainable weight/bias tensors are stored as FP32, including raw master weights; EMA is a separate FP32 copy excluded from the trainable total. Production uses CUDA BF16 autocast for Linear GEMMs. Learned projections are cast to FP32 before geometric products/reductions, local-frame products and final centering. kNN squared distances/tie sorting use FP64. Frames, torsions, chirality, vector normalizations and graph features use FP32.', '',
    'The CPU forward uses BF16 autocast to observe activation shapes and dtypes without GPU access. Those are CPU observations; CUDA policy is audited from source rather than probing the live trainer. CUDA autocast runs LayerNorm in FP32, including its FP32 affine tensors. CPU LayerNorm preserves its input dtype: the first fine norm has BF16 input/output in this trace; later residual H is FP32. The installed Torch 2.13 `include/ATen/autocast_mode.h` lists layer_norm/native_layer_norm under AT_FORALL_FP32. [PyTorch AMP documentation](https://docs.pytorch.org/docs/2.14/amp.html#cuda-ops-that-can-autocast-to-float32) explains the general CUDA policy; the installed header is the exact-version evidence recorded in JSON.', '',
    'All 16 TypedBlocks and their descendant layers are inside non-reentrant checkpoint regions in gradient-enabled training. Checkpointing requires training=True, the enabled configuration flag, and gradients enabled. It is correctly inactive in this no-gradient forward. Embeddings, mean pooling, decoder Broadcasts and heads are not checkpointed.', '',
    '## Synthetic activation trace','',
    'B=1, N=500, timestep 50; seed 150150500. Both current and previous coordinates are independent synthetic Gaussian FP32 tensors. Self-conditioning availability is true to exercise both graphs. Packed typed states omit the outer batch axis. H/V/P/A mean scalar/polar/pseudoscalar/axial. Raw branch intermediates are reconstructed from projection hook outputs without rerunning learned layers.', '',
    '| Stage | Observed shapes and dtypes |','|---|---|']
for key,value in activations.items():
    lines.append(f'| `{key}` | `{json.dumps(value,sort_keys=True)}` |')
lines += ['','## Synthetic directed graphs','',
    '| Scale | Nodes | Polymer edges | Current edges | SC edges | Deduplicated edges | i/j | Features | Unit pairs |',
    '|---|---:|---:|---:|---:|---:|---|---|---|']
for g in graph_records:
    lines.append(f'| {g["stage"]} | {g["coordinates"]["shape"][0]} | {g["polymer_edges"]} | {g["current_edges"]} | {g["self_conditioning_edges"]} | {g["deduplicated_edges"]} | {g["indices_i"]["shape"]} int64 | {g["edge_features"]["shape"]} FP32 | {g["unit_vectors"]["shape"]} FP32 |')
lines += ['',
    'Polymer/current/SC boolean provenance masks have shape [N,N]; directed COO index pairs have [2,E] (the actual graph returns two separate [E] int64 vectors). Fine current kNN indices are [500,24], SC [500,16]; medium [125,24]/[125,16]. Edge union size depends on this specific synthetic geometry, not learned weights. Coarse current connectivity is all 32×31=992 off-diagonal directed pairs; SC and polymer are subsets.', '',
    'The conceptual coarse dense attention matrix is [32,32] (one head, diagonal excluded). Implementation does **not** allocate that attention matrix: query/key are [32,64], logits/normalized weights are [992], scatter maxima/totals are [32]. Medium uses the same edge-wise softmax on its sparse union. RoPE runs in FP32 on the BF16 query/key projections.', '',
    '## Exact production settings','', '```json',json.dumps(production,indent=2),'```','',
    'SNR bins are equal-sized rank partitions of the frozen logSNR values, ordered from low to high SNR, not equal-width logSNR intervals. The full timestep membership is recorded above and in JSON. Geometry coefficients are frozen calibration values. Structural objectives are equal-protein means; the distance coefficient applies to the mean of L1/L2/L3. Geometry loss gates and the reliability-modulated output/parity gate are distinct.', '',
    '## Every registered module','',
    'Recursive counts overlap parent/child rows and must not be summed. Direct counts sum to the total. Weight/bias tensor names and shapes appear in the next table; JSON also links each module to its direct tensor names and complete input/output hook metadata. Functional stages are in the disjoint accounting table.', '',
    '| Module | Type | Direct | Recursive | % total (recursive) | In checkpoint region |',
    '|---|---|---:|---:|---:|---|']
for m in modules:
    lines.append(f'| `{m["name"]}` | {m["type"]} | {m["direct_parameters"]:,} | {m["recursive_parameters"]:,} | {m["percentage_of_total"]:.8f}% | {m["checkpoint_region"]} |')
lines += ['','## Every trainable tensor','',
    '| Tensor | Shape | Elements | Production storage | Operation dtype | Checkpoint region |',
    '|---|---|---:|---|---|---|']
for t in tensor_inventory:
    lines.append(f'| `{t["name"]}` | {t["shape"]} | {t["parameters"]:,} | FP32 | {t["production_operation_dtype"]} | {t["checkpoint_region"]} |')
lines += ['','## Buffers','', '```json',json.dumps(report['buffers'],indent=2),'```','',
    '## Noninterference and reproducibility','',
    f'Supervisor PID {after_owner["pid"]} and trainer PID {after_state["pid"]} remained the same; observed trainer updates {before_state["update"]} → {after_state["update"]}. CUDA was never initialized in this audit. No signal, trainer memory inspection, production optimizer access, checkpoint load/write or scientific source/config/calibration change occurred. The active trainer continues its own ordinary updates and checkpoint writes.', '',
    'Only this documentation, its CPU report generator, and `ARCHITECTURE_PARAMETER_AUDIT.json` are audit outputs. The generator is under docs so it is outside Runtime.contract source enumeration and cannot invalidate production certificates. The generated runtime JSON is not committed.', '',
    'Reproduce the CPU report using the production Python environment:', '',
    '```bash','/home/simostocco/miniforge3/envs/proteingen/bin/python -B docs/e015_architecture_parameter_audit.py','```','',
    'The generator stages JSON at `/tmp/ARCHITECTURE_PARAMETER_AUDIT.json`; publishing to the artifact root is a separate report-only write. It never calls Runtime initialization, GPU leases or production entry points.']
(WORK/'docs/e015_architecture_parameter_audit.md').write_text('\n'.join(lines)+'\n')
Path('/tmp/ARCHITECTURE_PARAMETER_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(total=total,groups=len(groups),modules=len(modules),tensors=len(tensor_inventory),
    graphs=[{k:g[k] for k in ['stage','deduplicated_edges']} for g in graph_records],
    trainer_updates=[before_state['update'],after_state['update']],cuda_initialized=False)),flush=True)
