#!/bin/bash
cd Code && python demo.py;
cd ../Report && pdflatex report.tex && cp report.pdf ../Results/;
