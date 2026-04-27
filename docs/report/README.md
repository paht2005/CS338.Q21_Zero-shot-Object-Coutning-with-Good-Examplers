# CS338.Q21 Report — `Report.pdf`

This directory builds `Report.pdf` from `main.tex` using `latexmk` + XeLaTeX.

## Prerequisites

- **TeX Live 2024+** (or BasicTeX on macOS — see below for extra packages)
- **`pygmentize`** (Python's Pygments) — required if you re-enable `minted` (currently the report uses `lstlisting` for code listings, so this is optional)
- **`latexmk`** (ships with TeX Live)

## Required LaTeX packages

The preamble of `main.tex` pulls in the following (most ship with full TeX Live;
those marked with `[basic]` must be installed manually if you only have macOS
**BasicTeX**):

| Package        | Purpose                                            | BasicTeX? |
|----------------|----------------------------------------------------|-----------|
| `tocloft`      | Custom CHƯƠNG~N entry in TOC                       | `[basic]` |
| `placeins`     | `\FloatBarrier`                                    | `[basic]` |
| `cleveref`     | `\cref`                                            | `[basic]` |
| `mathtools`    | math tweaks (loads `amsmath`)                      |           |
| `amssymb`      | `\mathbb{R}`                                       |           |
| `siunitx`      | numbers / units                                    | `[basic]` |
| `caption` / `subcaption` | figure captions, sub-figures             |           |
| `wallpaper`    | corner-image on the title page                     | `[basic]` |
| `nomencl`      | nomenclature support                               | `[basic]` |
| `fancyhdr`     | running headers                                    |           |
| `listings`     | source-code listings (bash demo block)             |           |
| `multirow` / `adjustbox` / `booktabs` / `colortbl` | tables          | `[basic]` (adjustbox) |
| `xurl`         | URL line-breaking                                  | `[basic]` |
| `fontspec` / `polyglossia` | XeLaTeX font + Vietnamese language     |           |
| `tex-gyre`     | `texgyretermes` (main body font)                   | `[basic]` |
| `pgfplots`     | charts                                             | `[basic]` |
| `lipsum`       | placeholder filler                                 | `[basic]` |

### macOS BasicTeX setup

```bash
# One-time: create a writable user tree so you do not need sudo
mkdir -p $HOME/.texlive-user
tlmgr --usermode --usertree=$HOME/.texlive-user init-usertree

# Install the missing packages
tlmgr --usermode --usertree=$HOME/.texlive-user install \
    tocloft placeins cleveref siunitx wallpaper nomencl \
    fvextra adjustbox collectbox xstring xurl pgfplots lipsum tex-gyre

# Make XeLaTeX find them on every build
export TEXMFHOME=$HOME/.texlive-user
```

Add the `export TEXMFHOME=...` line to `~/.zshrc` (or `~/.bashrc`) if you want
this to persist across shells.

> Note: `minted` is **not** relocatable into a user tree; the report therefore
> uses `lstlisting` for the single bash demo block. If you want syntax-coloured
> code listings via `minted`, install it system-wide with
> `sudo tlmgr install minted` and switch the `\begin{lstlisting}` block back to
> `\begin{minted}{bash}`.

## Build

```bash
cd docs/report
make pdf            # produces Report.pdf
make watch          # continuous rebuild on save (latexmk -pvc)
make clean          # remove aux/log files but keep Report.pdf
make distclean      # also remove Report.pdf
```

Under the hood `make pdf` runs:

```text
latexmk -xelatex -interaction=nonstopmode -halt-on-error -shell-escape \
        -file-line-error -jobname=Report main.tex
```

`-shell-escape` is required only for re-enabling `minted`; the current report
builds successfully without it as well.

## Output

- `Report.pdf` — final 26-page bilingual (Vietnamese) report for CS338.Q21
- `Report.log` / `Report.aux` / `Report.toc` / `Report.lof` / `Report.lot` —
  intermediate files, regenerated on each build
- `figures/` — image assets (see [figures/README.md](figures/README.md) for
  provenance and regeneration steps)
- `logos/` — UIT logos and title-page background

## Troubleshooting

- **`tocloft.sty not found`** → run the `tlmgr --usermode ...install` command
  above, then `export TEXMFHOME=$HOME/.texlive-user`.
- **`The font "TeX Gyre Termes" cannot be found`** → install the `tex-gyre`
  package; the preamble already references the `.otf` files by name so a system
  font registration is not required.
- **`Object @page.1 already defined`** (xdvipdfmx warning) → harmless; caused
  by the `wallpaper` background on the title page.
- **`Annotation out of page boundary`** (xdvipdfmx warning) → harmless; long
  hyperlink rendered inside an overlong table cell.
