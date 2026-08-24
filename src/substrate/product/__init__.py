"""Post-Odyssey product foundation.

This package is deliberately separate from the research campaigns.  It records
portable entity state and plans controlled capability use; it does not execute
tools, browse, fetch sources, or grant an entity open-world authority.
"""

from substrate.product.contracts import PRODUCT_SCHEMA_VERSION, ProductRefused

__all__ = ("PRODUCT_SCHEMA_VERSION", "ProductRefused")
