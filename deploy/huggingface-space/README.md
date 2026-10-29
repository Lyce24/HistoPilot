---
title: HistoPilot
emoji: 🔬
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
short_description: Read-only synthetic demo of a pathology MIL workspace
license: mit
---

# HistoPilot demo

[HistoPilot](https://github.com/Lyce24/HistoPilot) is a research workspace for computational pathology. It takes a slide table to cross-validated, compared and applied multiple instance learning (MIL) models, and keeps every step frozen, versioned and reproducible.

This Space runs its synthetic BLCA demo: a bladder-cancer grading study in which every record, label, score and curve is generated. It is read-only. You can open every stage, from datasets to models applied to new cohorts, but nothing is saved and no compute runs. To work with your own slides, install HistoPilot on your workstation; see the [quick start](https://github.com/Lyce24/HistoPilot#quick-start).

The container installs a released HistoPilot wheel and runs `histopilot serve --demo`, which serves the browser UI and the demo's reads and refuses everything else.
