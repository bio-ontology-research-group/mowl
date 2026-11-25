# System description of mOWL 3

`mowl3.tex` describes the design of mOWL 3: methods as combinations of exchangeable parts,
the method catalogue, the benchmark set with published numbers and recipes, and the
reproduction study. `mowl3.pdf` is the compiled document.

The numbers in it are those of the documentation at the time of writing (catalogue and
benchmark counts, `docs/source/benchmarks/three_way.csv`, the reproduction page).

To compile, the Springer LNCS class (`llncs.cls`, `splncs04.bst`, in TeX Live as the
package `llncs`) is needed:

```bash
pdflatex mowl3 && bibtex mowl3 && pdflatex mowl3 && pdflatex mowl3
```
