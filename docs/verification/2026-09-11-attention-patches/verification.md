# Highest-attention patch inspection — 2026-09-11

Model interpretation now pairs numbered Top 10 / Top 20 slide boxes with ranked original-color patch images. Selecting a box or card selects the same patch, focuses its slide location and shows the original crop, level-0 coordinates, footprint, normalized weight and percentile. Desktop displays the slide and crop side by side; mobile selection brings the crop into view and **Show patch location** returns to the slide.

## UI verdict

The two linked views answer both questions a user needs: where the model concentrated attention and what those image regions actually look like. Rank numbers connect boxes and cards. **Show ranked boxes** controls marker visibility independently of the heatmap. Existing ordinary patch clicks also open original crops. Keyboard activation, focus outlines and offscreen-marker filtering keep the controls usable without pointer precision.

Rankings cover the entire slide and remain independent of zoom, viewport limits and heatmap percentile filtering. They use the selected ensemble mean or member. Refit predictors expose their own single map. Equal weights use the original patch index as a deterministic tie-breaker; equal-weight ranks do not imply different importance. Overlapping or identical-coordinate rows are retained in the ranked cards, where they can be selected individually. Small slides show the available number of patches rather than inventing empty entries.

Member and slide changes isolate queries and clear selections. Ranking or attention verification errors hide stale selection evidence; ranking and crop errors have explicit retries. Crop images load lazily and share their authenticated cache between cards and the inspector. The inspector contains the original image without attention colors and preserves its aspect ratio.

Screenshots: [desktop location and crop](selected-desktop.png), [ranked Top 20 patches](top20-cards.png), [mobile crop](selected-mobile.png).

## Backend verdict

`GET /api/v1/projects/{project}/interpretations/{study}/slides/{slide}/attention/top` accepts the selected `member` and a `limit` from 1 to 20. The response contains whole-slide ranking metadata and patches ordered by descending raw normalized weight, then ascending original patch index. Each has a one-based rank.

Modern outputs use the authenticated N×4 numeric array. The reader scans fixed blocks of 65,536 rows and merges only bounded candidates, preserving exact tie order without loading millions of JSON objects. It validates all scanned coordinates and weights, closes mappings after use, and invalidates checksum verification on a changed full file stamp or receipt. Ranking never renormalizes the model output.

`GET .../patches/{index}/image` accepts a map member and a bounded image size. It looks up the authenticated patch row, reads the frozen original slide footprint, clips only at slide boundaries and checks that the source slide has not changed. The client cannot substitute arbitrary crop coordinates. Integer origins, fractional footprints, edge patches, unavailable indices and malformed or changed artifacts have regression coverage.

Legacy JSON maps remain supported within the historical 64 MiB writer/viewer limit. A compact bounded cache retains authenticated top rows, while a shared lock serializes full legacy parsing. It does not retain complete parsed maps or hold the parse lock during slide rendering. Original JSON exports remain available under their existing export limit.

## Verification

| Check | Result |
| --- | --- |
| Full frontend suite | 320 passed across 48 files |
| `test_attention_top.py`, `test_interpretation.py`, `test_attention_top_api.py`, `test_interpretation_gallery.py` | 76 passed |
| Architecture import-boundary tests | 3 passed |
| Final legacy-cache scalar-row hardening | 6 affected tests passed |
| TypeScript, Ruff and production build | Passed |
| [Browser interaction checks](browser-checks.json) | 34 retained checks |
| [Genuine attention and crop comparisons](real-attention-crops.json) | 60 matching ranks/crops across ensemble mean and both distinct members |
| Local wheel build and source/frontend byte verification | Passed |

The browser ran actual React components in a standalone local HTML fixture with controlled API responses. A synthetic 30-patch slide used genuine ABMIL checkpoint outputs; a seven-patch fixture covered fewer-than-ten behavior. Checks covered Top 10/20, member and refit ranking, pixel-exact crops, numbered-box and card selection, keyboard activation, ordinary heatmap clicks, delayed requests, verification failures and retries, desktop side-by-side visibility and mobile navigation. No unexpected network requests or browser runtime errors were observed.

A direct two-million-row ranking check found the highest-weight patch at index 1,999,999, beyond the viewport display limit. The local measurement was approximately 0.20 seconds including first checksum verification, with 2.01 MiB of traced scratch allocations alongside the 61.04 MiB mapped array. This is a local synthetic measurement, not a production WSI throughput guarantee.

The retained local fixture is `/tmp/histopilot-top-attention/fixture`. API tests used in-process clients; development did not start or restart HistoPilot. The packaged wheel includes the final API, verified-array reader and rebuilt frontend assets; the build retains the existing large-JavaScript-chunk advisory. Representative vendor-slide and storage performance still depends on the deployment's OpenSlide support and hardware. Attention remains class-independent ABMIL pooling importance, and the crop is the original slide region rather than necessarily the extractor's resized model input.
