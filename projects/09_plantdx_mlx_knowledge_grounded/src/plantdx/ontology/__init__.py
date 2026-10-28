"""Ontology package (component A): DKB → caption ontology derivation.

Models live in :mod:`plantdx.ontology.models`.
"""

from __future__ import annotations

from plantdx.ontology.builder import OntologyBuilder
from plantdx.ontology.concept_schema import ConceptSchemaLoader
from plantdx.ontology.models import Ontology

__all__ = ["ConceptSchemaLoader", "Ontology", "OntologyBuilder"]
