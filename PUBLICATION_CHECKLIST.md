# Before public release

1. Upload the separate private assets package to the journal submission system for authorized reviewers. The public GitHub repository contains no processed data, model files, or row-level predictions.
2. The authors report that the SSA implementation is already public on GitHub. Add its upstream repository link if you want readers to trace that source. Add the author list, manuscript citation or DOI, and contact email when available. No `CITATION.cff` is included because these details were not supplied.
3. Decide whether to add a software license and, if so, which one. GitHub does not require a license to make a repository public, but without one, reuse rights are limited. Do not apply one license to third-party material unless its terms allow that.
4. Run the README audits with the paired private assets package and check `REPRODUCIBILITY_STATUS.md`, including the current M5 rebuild. The M5 fit and its 0.880729 reference are new to this version.
5. After uploading, link the final GitHub commit or archived release DOI from the manuscript Code Availability statement and freeze a release tag for the submitted version.

Do not describe M2–M5 as one controlled ablation with identical model searches. They share a 9/24 panel, but M2–M4 are historical runs and M5 is a new weighted XGBoost + SVR fit; the stage 3 provenance file gives the exact mapping and search budgets.
