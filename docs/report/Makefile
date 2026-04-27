# Build the CS338.Q21 project report.
#
# Requires a TeX Live distribution with XeLaTeX, latexmk, polyglossia and the
# `vntex` packages installed. On macOS:
#
#     brew install --cask mactex
#     sudo tlmgr update --self && sudo tlmgr install latexmk polyglossia vntex \
#         minted pgfplots wallpaper xurl
#
# On Ubuntu / WSL / Debian:
#
#     sudo apt install texlive-xetex texlive-lang-vietnamese \
#         texlive-fonts-extra texlive-pictures texlive-science latexmk \
#         python3-pygments

MAIN     := main
ENGINE   := xelatex
LATEXMK  := latexmk
LATEXMK_FLAGS := -$(ENGINE) -interaction=nonstopmode -halt-on-error -shell-escape -file-line-error

REPORT_PDF := Report.pdf

.PHONY: all pdf clean distclean watch

all: pdf

pdf: $(REPORT_PDF)

$(REPORT_PDF): $(MAIN).tex $(wildcard *.bib) $(wildcard figures/*) $(wildcard logos/*)
	$(LATEXMK) $(LATEXMK_FLAGS) -jobname=Report $(MAIN).tex

watch: ## Continuously rebuild on changes
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc -jobname=Report $(MAIN).tex

clean: ## Remove auxiliary build artefacts but keep Report.pdf
	$(LATEXMK) -c -jobname=Report
	rm -rf _minted-*

distclean: clean ## Also delete the generated Report.pdf
	$(LATEXMK) -C -jobname=Report
	rm -f $(REPORT_PDF)
