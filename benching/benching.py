#!/usr/bin/env python3
"""
Entrypoint for threat precursor forecasting benchmarks across attack types.
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from benchmark_attack_forecasting import main

if __name__ == "__main__":
    main()
