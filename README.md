<h1 align="center">
<img src="logo.png" width=200px></img>
</h1>

Periodica is a
C++ based Python library for analyzing the topological structures
of a periodic set. It can compute how the points connect with
each other (captured by the Delaunay triangulation) at different
length scales (encoded in the periodic merge tree and the
topological descriptors). For more information, please check our 
[CECAM workshop poster](cecam_poster.pdf).

## Build

To build Periodica from source, please install [bazelisk](https://github.com/bazelbuild/bazelisk), and simply run

```
make
```

## Web UI

A browser frontend (edit lattice/points/weights, visualize the periodic Delaunay
with a filtration slider, and view barcode/diagram/image descriptors) lives in `web/`:

```
make web                          # FastAPI backend on :8000 (serves web/frontend/dist if built)
cd web/frontend && npm install && npm run dev   # dev server on :5173
```

## References

Periodica is based on the following research papers:

```
@misc{EH2024,
    title         = {Merge Trees of Periodic Filtrations}, 
    author        = {Herbert Edelsbrunner and Teresa Heiss},
    year          = {2024},
    eprint        = {2408.16575},
    archivePrefix = {arXiv},
    primaryClass  = {math.AT},
    url           = {https://arxiv.org/abs/2408.16575}, 
}

@InProceedings{ORT2020,
    title     = {Generalizing CGAL Periodic Delaunay Triangulations},
    author    = {Georg Osang and Mael Rouxel-Labb\'{e} and Monique Teillaud},
    booktitle = {28th Annual European Symposium on Algorithms (ESA 2020)},
    year      = {2020},
    pages     = {75:1--75:17},
    volume    = {173},
    address   = {Dagstuhl, Germany},
    URL       = {https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2020.75},
    doi       = {10.4230/LIPIcs.ESA.2020.75},
}
```
