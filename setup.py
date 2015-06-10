#!/usr/bin/env python
from setuptools import setup, find_packages

setup (
    name = "katcal",
    version = "trunk",
    description = "MeerKAT calibration pipeline",
    author = "Laura Richter",
    author_email = "laura@ska.ac.za",
    packages = find_packages(),
    scripts = [
        "scripts/reduction_script.py",
	"scripts/run_cal.py",	
        "scripts/run_katcal_sim.py",
        "scripts/sim_l1_receive.py",
        "scripts/sim_h5_ts.py",
        "scripts/sim_h5_stream.py"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    platforms = [ "OS Independent" ],
    keywords="kat kat7 meerkat ska",
    zip_safe = False,
    # Bitten Test Suite
    #test_suite = "katfile.test.suite",
)
