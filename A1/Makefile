all: plots/task2.pdf plots/task3.pdf 

plots/task2.pdf: code/task2.py | plots
	python code/task2.py

plots/task3.pdf: code/task3.py | plots
	python code/task3.py

plots:
	mkdir plots

clean:
	rm -r plots
	rm -r data

.PHONY: all clean