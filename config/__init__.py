import importlib, pkgutil, inspect

ENV_REGISTRY  = {}   # "32node_24tm" → dict
ALGS_REGISTRY = {}   # "ls2ic" → dict
CONTROLLER_REGISTRY = {}

def _scan_pkg(subpkg_name, target_dict):
    pkg = importlib.import_module(f"config.{subpkg_name}")
    for modinfo in pkgutil.iter_modules(pkg.__path__):
        if modinfo.name.endswith("_config"):
            mod = importlib.import_module(f"config.{subpkg_name}.{modinfo.name}")
            if hasattr(mod, "config"):
                key = modinfo.name.replace("_config", "")
                target_dict[key] = mod.config

# 執行掃描
_scan_pkg("env",  ENV_REGISTRY)
_scan_pkg("algs", ALGS_REGISTRY)
_scan_pkg("controller", CONTROLLER_REGISTRY)

def get(env: str, alg: str, ctrl: str):
    return ENV_REGISTRY[env], ALGS_REGISTRY[alg], CONTROLLER_REGISTRY[ctrl]
