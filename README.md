# LabProc & Tacit

**Anonymous repository for NeurIPS 2026 Datasets & Benchmarks submission.**

This repository will host the benchmark, evaluation code, and adapted model weights for *LabProc: A Benchmark for Laboratory Procedure Understanding via Domain-Adapted Video World Models*.

## Status

This is a placeholder repository created at the abstract submission deadline. The full release will be available at the paper submission deadline and will include:

- **LabProc benchmark** — six tasks for laboratory procedure understanding: physical state classification (PSC), procedure causal reasoning (CCR), transition error detection (TED), visual state discrimination (VSD), visual continuation reasoning (TED-Visual), and same-state ordering (Same-State CCR).
- **Tacit** — a domain-adapted V-JEPA-2.1 encoder trained on 155 hours of curated chemistry footage (1,037 videos spanning organic purification, polymerase chain reaction, and Western blot procedures).
- **Evaluation harness** — scripts to reproduce all reported results, including comparisons against base V-JEPA-2.1 (305M parameters) and a frontier vision-language model.
- **Annotations and splits** — task-level labels and standardized train/eval splits.

## Planned repository structure

```
labproc/
├── benchmark/
│   ├── psc/                  # Physical state classification
│   ├── ccr/                  # Procedure causal reasoning
│   ├── ted/                  # Transition error detection
│   ├── vsd/                  # Visual state discrimination
│   ├── ted_visual/           # Visual continuation reasoning
│   └── same_state_ccr/       # Same-state ordering
├── tacit/
│   ├── model/                # V-JEPA-2.1 adaptation code
│   ├── inference.py          # Run Tacit on a video
│   └── weights/              # Pointer to model weights (Hugging Face)
├── eval/
│   ├── run_eval.py           # Reproduce headline results
│   └── compose.py            # V+L composition (Tacit + VLM)
├── data/
│   └── README.md             # Data access instructions
└── README.md
```

## Reproducing the headline results

Once released, the following will reproduce the key numbers in the paper:

```bash
# Physical state classification (10-class)
python eval/run_eval.py --task psc --model tacit
# Expected: 34.1%
python eval/run_eval.py --task psc --model vjepa-base
# Expected: 25.6%

# Visual state discrimination
python eval/run_eval.py --task vsd --model tacit
# Expected: 56.1%

# PSC-Combined (V+L composition)
python eval/compose.py --vision tacit --language claude-opus
# Expected: 64.5%
```

## License

The benchmark, code, and adapted model weights will be released under a permissive license (TBD at camera-ready) suitable for academic and non-commercial use.

## Contact

Author identity withheld for double-blind review. Correspondence will be enabled at camera-ready.
