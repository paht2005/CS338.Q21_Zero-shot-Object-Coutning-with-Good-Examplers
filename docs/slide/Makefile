TEX = xelatex
SRC = main.tex
OUT = main.pdf

all: $(OUT)

$(OUT): $(SRC)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fdb_latexmk *.fls *.synctex.gz

distclean: clean
	rm -f $(OUT)

.PHONY: all clean distclean
