"""What every published-circuit importer returns, so exporters need not know the circuit."""

from dataclasses import dataclass

from cirqloom_analysis.schema.bundle import Circuit, Edge, Membership


@dataclass(frozen=True)
class PublishedCircuit:
    """One published circuit, ready to add to a bundle graph."""

    circuit: Circuit
    memberships: list[Membership]
    edges: list[Edge]
    provenance: str
    """One line for the bundle's provenance: source, version, and where it was verified."""
