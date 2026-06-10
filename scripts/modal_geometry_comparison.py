"""
Head-to-head geometry-aware vs plain multi-view at K=8 views.

Geometry-aware should win here: 8 views from different azimuths are harder to
fuse by blind mean-pooling; explicit camera geometry disambiguates each view.

Usage:
  # Train both models in parallel (~45-90 min each)
  scripts/baselines/run_geometry_comparison.sh

  # Or one model
  modal run --detach scripts/modal_train_baseline.py \\
    --config configs/baselines/main_geometry_aware.yaml --compare

  # Visualize after training
  modal run scripts/modal_visualize_baselines.py --fetch-local
"""
