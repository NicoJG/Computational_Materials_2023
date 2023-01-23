all: plots/task2.pdf 

plots/task2.pdf: code/task2.py | plots
	python code/task2.py

plots:
	mkdir plots

clean:
	rm -r plots
	rm -r data

.PHONY: all clean