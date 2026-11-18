"""FlyLab: a reproducible, inspectable spiking simulation built on the MaleCNS v1.0 connectome.

Status: milestone M1 (dataset import). Contains the MaleCNS v1.0 importer (`flylab data ...`);
no neural engine or body backend yet. See docs/IMPLEMENTATION_SPEC.md.
"""

__version__ = "0.0.1"

#: Version of every on-disk / over-the-wire payload FlyLab writes. Bump on breaking change.
SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION", "__version__"]
