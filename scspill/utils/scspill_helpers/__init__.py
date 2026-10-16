"""Shared spillover layer, plus one subpackage per model.

The shared layer serves every model:

- :mod:`.config`: the configuration, ``SCSPILLConfig``;
- :mod:`.setup`: panel ingestion;
- :mod:`.containers`: the panel and effect containers;
- :mod:`.structures`: the result, ``SCSPILLResults``;
- :mod:`.plotter`: plotting.

Each model adds a subpackage with its own setup, estimation, inference,
pipeline, report and structures: :mod:`.sar` for Sakaguchi & Tagawa's
Bayesian SAR model, :mod:`.cd` for Cao & Dowd's model. The layout mirrors
mlsynth's ``spillsynth_helpers``.
"""
