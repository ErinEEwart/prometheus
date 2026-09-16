#!/usr/bin/env python3
"""pone-opt-make-event-library.py
Based on 01_basic_water_gpu
"""

import logging
import sys

logger = logging.getLogger(__name__)

try:
    from prometheus import Prometheus, config
except Exception:
    logger.exception(
        "Error importing Prometheus. "
        "Ensure the environment is activated and requirements are installed."
    )
    logger.info(
        "Hint: source scripts/activate.sh .prometheus_env && pip install -r requirements.txt"
    )
    sys.exit(1)

# Prefer CPU JAX when available
try:
    from jax.config import config as jconfig

    #jconfig.update("jax_enable_x64", True)
    #jconfig.update("jax_platform_name", "cpu")
except Exception:
    pass


def main():
    # Minimal runtime configuration
    config.run.run_number = 1
    config.run.random_state_seed = 1
    config.run.nevents = 10

    # Injection: minimal LeptonInjector settings
    config.injection.name = "LeptonInjector"
    config.injection.lepton_injector.simulation.is_ranged = False
    config.injection.lepton_injector.simulation.minimal_energy = 1e7-1 # GeV
    config.injection.lepton_injector.simulation.maximal_energy = 1e7 ## 1e7 monoenergetic @ 10 PeV
    print("Energy: ", config.injection.lepton_injector.simulation.maximal_energy)

    # Use the demo water geo shipped in resources (repo root path)
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parent.parent
    config.detector.geo_file = str(REPO_ROOT / "pone-optimize" / "single_string.geo")
    config.run.storage_prefix = str(REPO_ROOT / "output") + "/"

    config.run.verbosity = 'DEBUG'
    #config.run.summary_mode='debug'

    print("Initializing Prometheus (minimal)")
    prom = Prometheus()

    try:
        prom.sim()
    except Exception:
        logger.exception("Simulation error during prom.sim()")
        logger.info(
            "Hint: ensure resources (LeptonInjector, model files) "
            "are available and config is correct."
        )
        sys.exit(1)

    print("Simulation completed successfully")


if __name__ == "__main__":
    main()
