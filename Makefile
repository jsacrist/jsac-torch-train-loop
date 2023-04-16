.ONESHELL:
.PHONY: build, build-wheel, clean, test, docs, docs-server

build:
	python3 setup.py sdist

build-wheel:
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

release-test: clean build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
	