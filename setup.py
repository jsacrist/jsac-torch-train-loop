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
print(f"Installing {packages} version {version}.")

setup(
    name=f"{NAMESPACE}_{PACKAGE}",
    version=version,
    cmdclass=cmdclass,
    description="",
    long_description="",
    long_description_content_type="text/markdown",
    author="Jorge Sacristan",
    author_email="j.sacris@gmail.com",
    url="https://github.com/jsacrist/jsac-torch-train-loop.git",
    project_urls={
        "Bug Tracker": "https://github.com/jsacrist/jsac-torch-train-loop/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=packages,
    include_package_data=True,
    install_requires=["torch", "tqdm"],
    # entry_points={
    #     "console_scripts": []
    # },
    zip_safe=False,
    python_requires=">=3.6",
)
