PREFIX = poster-mainfile
TEXFILE = $(PREFIX).tex
BIBFILE = $(PREFIX).aux
LATEX = pdflatex
BIBTEX = bibtex
TRASH = *.aux *.log *.dvi *.ps *.nav *.out *.snm *.backup *.bak *.toc *~ *.bbl *.blg *.toc

short: 
	$(LATEX) -halt-on-error $(TEXFILE)

all: 
	$(LATEX) -halt-on-error $(TEXFILE)
	$(BIBTEX) $(BIBFILE)
	$(LATEX) -halt-on-error $(TEXFILE)
	$(LATEX) -halt-on-error $(TEXFILE)

clear:
	rm -rf $(TRASH)
