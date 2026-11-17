# Third-party source and external assets

The project's MIT license covers original contributions, subject to the notices below. It does not replace upstream conditions or license external weights, data or outputs.

| Local module | Source/provenance | Notice |
|---|---|---|
| `src/protein_distance_diffusion/models/e007_proteinmpnn_ca.py` | Adapted CA-only inference components from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57); the module lists the source-to-local mapping. | Copyright (c) 2022 Justas Dauparas; [MIT notice](licenses/ProteinMPNN-MIT.txt), retained from the [pinned upstream license](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/LICENSE). |
| `src/protein_distance_diffusion/models/e007_progen2.py` | Compatibility implementation following the official ProGen2 checkpoint layout; pinned checkpoint revision is recorded in the module. | Conservative attribution to Salesforce's [ProGen source](https://github.com/salesforce/progen), copyright (c) 2022 Salesforce.com, Inc.; [BSD-3-Clause notice](licenses/ProGen-BSD-3-Clause.txt) from [upstream](https://github.com/salesforce/progen/blob/main/LICENSE.txt). The exact checkpoint license must be verified separately before acquiring or redistributing weights. |

No external checkpoint or vendor model payload is added by this integration. ESM/Transformers and other installed packages retain their upstream licenses; imports are not a redistribution of their source. No affiliation or endorsement by upstream authors is implied. Dataset terms and raw evidence redistribution require separate review. This inventory is a source/provenance audit, not a certification of every historical artifact's license.
