# -*- coding: utf-8 -*-
from setuptools import setup

NAMESPACE = "jsac"
PACKAGE = "torch_train_loop"


# Loads _version.py module without importing the whole package.
def get_version_and_cmdclass(package_name):
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(package_name, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.get_cmdclass(NAMESPACE)


version, cmdclass = get_version_and_cmdclass(f"{NAMESPACE}/{PACKAGE}")
packages = [f"{NAMESPACE}.{PACKAGE}"]
print(f"Running setup.py for {packages} version [{version}]")

setup(
    name=f"{NAMESPACE}_{PACKAGE}",
    version=version,
    cmdclass=cmdclass,
)
