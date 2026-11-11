# SpaCoEx Workflow: Standalone TikZ

## Overleaf

1. Upload `spacoex_workflow_tikz_overleaf.zip` as a new Overleaf project, or upload its `main.tex` into a blank project.
2. Select pdfLaTeX as the compiler and `main.tex` as the main document.
3. Recompile. The standalone document produces one tightly cropped figure.

Only `main.tex` is required. It uses the standard `standalone`, `tikz`, `fontenc`, `helvet`, and `amsmath` packages. No shell escape, raster images, external data, or Python is needed in Overleaf.

## Contents

- Five numbered panels, arrows, captions, and labels.
- 252 tissue-map markers, 77 Gaussian-neighborhood markers, 217 PCA markers, and 245 spatial-cluster markers, plus the center/legend markers.
- Every heatmap cell is a colored vector rectangle.
- The Gaussian weight curve is drawn from its mathematical expression.
- Heatmaps representing correlations are symmetric with a unit-correlation diagonal. The selected heatmap is the matching principal submatrix of the large heatmap.

The dot centers, approximate ellipse shapes, and colors are traced from the reference illustration, not from experimental measurements. Text and matrix drawings are reconstructed, so the vector drawing is not a pixel-for-pixel conversion of the AI-generated bitmap.

The source is organized into commented sections. Dot coordinate lists are static and editable. Change `\UNIT` near the beginning to resize the geometry; adjust the `title`, `body`, `caption`, and `gene` font styles separately if needed.

## Verification

The generated source was checked for balanced braces, matching environments, expected vector-marker counts, and absence of external-image dependencies. Local compilation could not be verified: the desktop LaTeX compiler refused to start because of a Windows sandbox error, and no terminal TeX compiler was installed.

`build_vector_workflow.py` and `tracing_check.png` are local reconstruction aids. Neither is included in the Overleaf ZIP or required to compile the figure.
