.ONESHELL:
.PHONY: build-src, build-wheel, clean, test, docs, docs-server

build-src:
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

docs-html:
	rm -rf docs/source/auto_examples/*
	cd docs; make clean html

docs-server:
	python3 -m http.server 8000 --directory=./docs/build/html/

release-test: clean build-wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*.whl
	echo "After uploading a release to the TEST repository, you can try installing your"
	echo "package by running the following command:"
	echo ">>> pip3 install torch-train-loop --extra-index-url=https://test.pypi.org/simple/"
	