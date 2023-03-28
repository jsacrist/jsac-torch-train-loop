.ONESHELL:
.PHONY: clean, build, test, docs

build:
	python3 setup.py bdist_wheel

clean:
	rm -rf build/
	rm -rf dist/
	find . -iname "__pycache__" -exec rm -rv {} +
	find . -iname "*.egg-info" -exec rm -rv {} +

test:
	pytest

docs:
	rm -rf docs/source/auto_examples/*
	cd docs; make clean html

docs-server:
	python3 -m http.server 8000 --directory=./docs/build/html/
