from .qafm    import QAFM
from .badnets import BadNets
from .ftrojan import FTrojan
from .blended import Blended


def build_attack(method: str, cfg: dict):
    """
    공격 인스턴스 팩토리.

    Args:
        method: "qafm" | "badnets" | "ftrojan" | "blended"
        cfg:    해당 attack config dict
    """
    method = method.lower()
    if method == "qafm":
        return QAFM(**cfg)
    elif method == "badnets":
        return BadNets(**cfg)
    elif method == "ftrojan":
        return FTrojan(**cfg)
    elif method == "blended":
        return Blended(**cfg)
    else:
        raise ValueError(f"Unknown attack method: {method}")
