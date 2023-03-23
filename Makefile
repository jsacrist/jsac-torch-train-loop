.ONESHELL:
.PHONY: clean, build, test

build:
	python3 setup.py bdist_wheel

clean:
	rm -rf build/
	rm -rf dist/
	find . -iname "__pycache__" -exec rm -rv {} +
	find . -iname "*.egg-info" -exec rm -rv {} +

test:
	pytest