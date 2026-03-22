# Presentation Assets

Build the full slide package after training finishes:

```bash
python presentation/build_assets.py
```

Outputs land in `presentation/`:

- `figures/`: deck-ready PNGs
- `tables/`: metrics and patient-level alert summaries
- `deck_outline.md`: slide-by-slide story and speaker notes
- `rescuewindow_deck.pptx`: generated PowerPoint deck
- `summary.json`: headline metrics for quick reuse
