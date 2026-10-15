# Policy schema and governance semantics

Version: e016-governance/1. Validator: framework.validate_policy. The JSON instance and validator are hashed in the stage provenance manifest. Only PREPARATION_ONLY is accepted; changing a status string cannot grant production admission.

| Field | Meaning and constraint |
|---|---|
| policy_id / schema_version | Stable proposal identity and exact validator contract version |
| base_revision | Audited 40-hex source revision |
| evidence | Nonempty unique D: paths, 64-hex SHA256 and explicit role; lexical validation is not fresh file verification |
| relationships.exact_identity | HARD_CLOSURE identity invariant; no biological threshold |
| relationships.observation_ownership | HARD_CLOSURE ownership invariant; never split one identity's observations |
| relationships.similarity / fragment / construct | UNRESOLVED predicate and mode; no 30% or coverage threshold silently approved |
| relationships.source_context | UNRESOLVED safeguard; source co-occurrence is not ancestry evidence |
| shadow_policy | UNRESOLVED node participation; shadows are not mislabeled as a similarity edge |
| controls | Separate role model required; selected historical identities preserved; propagation scope UNRESOLVED |
| partitions | Soft 80/10/10 objective, seed42; denominator UNRESOLVED; this is not an assignment |
| evaluation | Unique/group/joint minima and length/method/context acceptance all null pending review |
| guarantees | Real data NOT_EVALUATED; no remote-homology elimination or production admission |

Relationship execution modes for a future reviewed schema are HARD_CLOSURE, AUDITED_CONTEXT_RESTRICTION and UNRESOLVED. A restriction needs its own versioned executable predicate, consumer/task, embargo/exclusion semantics, missing-metadata policy and verifier. None is implemented for real data. This preparation validator rejects resolving any ambiguous rule.

Control panel membership, ordinary development, reward development, final controls and training embargoes are distinct concepts. Panel membership must not imply entry-wide transitive exclusion by an undocumented default. Future role propagation and conflict resolution need approval. Toy reservation checks illustrate incompatible roles, not an adopted real control policy.

Provenance before real use must include the complete canonical policy hash, all source/mapping/relationship inputs, code and tool revisions, canonicalization policy, evidence completeness, observation ownership, prior split lineage, selected control identities and explicit claims/limitations. Directory names or a retained forest hash alone are insufficient policy identity. Source bytes must be revalidated without changing them; actual path resolution and source integrity are separate from lexical JSON validation.

Pre-allocation gates distinguish impossible hard constraints from unattainable soft targets and unproved feasibility. Necessary bounds cannot prove a component-packing solution. A future allocation must produce a deterministic whole-component witness and verify all hard constraints, task-specific eligible views and reported target deviations.

Certification requires independently verified sequence/observation ownership, reviewed biological predicates, declared context safeguards, auditable control roles, preserved TEST, adequate independent evaluation and source integrity. Nonempty views and balanced ratios are insufficient. Numerical acceptance remains a scientific choice.
