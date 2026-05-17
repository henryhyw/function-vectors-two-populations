# Auxiliary results

Out-of-band experiments run on top of the main per-cell pipeline at
`paper/results/{hierarchical,modular}/F0/`. Each subdirectory is one
experimental theme, and each file is one (model, task) cell payload.

All themes are produced by `paper/pipeline.ipynb`, which composes the
modules under `paper/code/`. The mapping between subdirs, modules and
notebook sections is:

| Subdir                              | Module / driver                                       | Notebook §  |
|-------------------------------------|-------------------------------------------------------|-------------|
| `sign_shuffle_n10k/`                | `code/group_lesion.sign_shuffle_null`                 | §5          |
| `per_source_dla/`                   | `code/per_source_dla.run`                             | §6          |
| `head_randomized_control/`          | `code/rule_outs.head_randomised_control`              | §8          |
| `rank1_vcascade_per_cell/`          | `code/rule_outs.rank1_share`, `code/rule_outs.v_cascade` | §8       |
| `v_shuffle_replication/`            | `code/rule_outs.v_shuffle`                            | §8          |
| `steering_transplant/`              | replacement-transplant (writer-only vs FV-mean) on $6$ main cells | App.~D.2 |
| `direction_decomp/`                 | `code/steering.build_steering_vectors` (OV geometry)  | App.~D.4    |
| `fv_overlap_todd/`                  | `code/dla.screen_fv_candidates` ∩ Todd top-K          | §2          |
| `vocab_transfer/`                   | `code/cross_template.run`, `cross_template.l11h4_solo_per_template` | §10 |
| `mechinterp_l11h4/`                 | `code/case_study.run`                                 | §12         |
| `cross_family/`                     | `code/scale_extension.run_one_cell`                   | §11         |
| `scale_extension/`                  | `code/scale_extension.run_one_cell`                   | §11         |

## Headline summaries (collected by `code/aggregate.py`)

- **Sign-shuffle null.** Observed `|W − C|` contrast lies outside the
  null 95% band in 5/6 main cells; mod-1.4B is the boundary case at
  `p_emp = 0.104`. Holm–Bonferroni at α=0.05 rejects in 5/6.
- **Per-source DLA.** 20/27 cancellers (74%) are content-driven across
  the six main cells; 7/27 are sink-driven (BOS or format-prefix).
  L11.H4 is ~100% content-driven on both Pythia-410M cells.
- **L11.H4 case study.** V-shuffle collapses cancellation by ~82%
  (hier-410M) / ~53% (mod-410M). OV singular spectrum: top-1 Frobenius
  share = 2.8% (no rank-1 plateau). Ablating the dominant upstream
  writer L10.H9 leaves L11.H4's DLA unchanged (95% CI on the diff
  includes 0).
- **V-shuffle replication.** On 8 additional cancellers spanning all
  six cells, the highest-magnitude canceller per cell is content-driven.
- **Cross-template transfer.** W/C labels from Pythia-410M transfer to
  vocabulary ICL (antonym, country-capital) in 3/4 directed pairs. The
  one failure is hier→antonym, where the canceller subgroup sign-flips
  to a super-writer (group Δℓ = −1.56 nats), and L11.H4 itself flips
  role from canceller to writer.
- **Scale + cross-architecture.** All 9 extension cells (Pythia 2.8B /
  6.9B / 12B; Qwen2.5-1.5B / 7B; GPT-2-medium) pass the signed-direction
  condition; 6.9B-mod and 12B-hier shift to PARTIAL (one CI grazes 0).
  Combined with the six main cells: 13/15 canonical, 2/15 partial,
  0/15 sign-flipped.
- **Transplant accuracy.** Writer-only ≤ FV-mean on accuracy in 6/6
  main cells (range −0.115 to −0.005). The gap tracks the canceller
  fraction in Todd's top-K (App. D.2).
- **Canceller ablation accuracy.** Zero-ablating cancellers lifts
  accuracy +2 to +7 pp directionally in 6/6 main cells; CI excludes 0
  on hier-410M (App. D.3).

## Reproducing one theme

```python
from code import io, prompts, config, group_lesion
model = ...   # HookedTransformer for the chosen cell
partition = io.load('hierarchical', 'validate', 'circuit_test', '410m')
eval_p = prompts.generate_batch('hierarchical', config.N_EVAL, base_seed=44)
null = group_lesion.sign_shuffle_null(model, eval_p, partition, n_perm=10_000)
io.aux_save('sign_shuffle_n10k', 'sign_shuffle_hierarchical_410m', null)
```

Each module exposes a single `run(...)` entry point that the notebook
calls; outputs are byte-stable for fixed seeds.
