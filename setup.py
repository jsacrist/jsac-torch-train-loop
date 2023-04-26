# -*- coding: utf-8 -*-
from setuptools import setup

NAMESPACE = "jsac"
PACKAGE = "torch_train_loop"
DIST_DIR = f"src/{NAMESPACE}/{PACKAGE}"


# Loads _version.py module without importing the whole package.
def get_version_and_cmdclass(package_name):
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "version", os.path.join(package_name, "_version.py")
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.get_cmdclass(package_name)


version, cmdclass = get_version_and_cmdclass(DIST_DIR)
print(f"Running setup.py for {DIST_DIR} version [{version}]")

setup(
    version=version,
    cmdclass=cmdclass,
)
