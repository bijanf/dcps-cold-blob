"""Importable draw_subpanel(ax) for the SAHEL pathway of Fig. 7 panel (c).

Re-exports the same function defined in compute_fig7_sahel.py so the
composing agent can `from dcps.scripts._subpanel_sahel import draw_subpanel`.
"""
from dcps.scripts.compute_fig7_sahel import draw_subpanel  # noqa: F401
