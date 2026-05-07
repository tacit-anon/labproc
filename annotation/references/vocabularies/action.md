# Action vocabulary — `action_tags` column

Multi-select. Apply all tags describing the human/instrument action(s) visibly happening in the frame. Comma-separated.

## Transfers / additions

| tag | description |
|---|---|
| `pouring` | liquid being poured from one vessel to another |
| `transferring` | substance being moved between vessels (catch-all when not strictly pouring) |
| `dispensing` | solvent or reagent being measured / dispensed from a stock bottle |
| `pipetting` | pipette being used |
| `loading_column` | sample being applied to top of a chromatography column |

## Filtration / separation

| tag | description |
|---|---|
| `filtering_gravity` | hot or cold gravity filtration through a fluted funnel |
| `filtering_vacuum` | Buchner funnel + filter flask under vacuum |
| `decanting` | liquid being separated by tipping off a settled phase |
| `extracting_lle` | active liquid-liquid extraction motion (shaking sep funnel) |
| `draining_layer` | stopcock open, lower phase being collected |

## Mixing / agitation

| tag | description |
|---|---|
| `stirring` | magnetic stirrer or stir rod actively mixing |
| `swirling` | flask being swirled by hand |
| `vortexing` | tube on a vortex mixer |
| `shaking` | shaker or by-hand rocking |

## Heating / cooling

| tag | description |
|---|---|
| `heating` | hot plate on, mantle on, or visible boiling |
| `refluxing` | active reflux: solvent boiling and condensing back |
| `distilling` | distillation apparatus actively running |
| `cooling_ice_bath` | flask in ice bath |
| `evaporating_rotovap` | rotovap actively concentrating |

## Measurement

| tag | description |
|---|---|
| `weighing` | sample on a balance |
| `measuring_volume` | graduated cylinder / volumetric flask in use |
| `titrating` | burette dripping into stirred sample |
| `tlc_spotting` | sample spots being applied to a TLC plate |

## Other

| tag | description |
|---|---|
| `assembling` | apparatus being put together (clamps, joints) |
| `cleaning` | rinsing / washing glassware |
| `observing_static` | nothing actively happening — static observation shot |

## Rules

- **Use `observing_static` only when no other action applies.** It's a fallback, not a default.
- Multiple actions are normal (e.g., `pouring, filtering_gravity` for hot filtration; `stirring, heating` for refluxing).
- Don't over-decompose: if it's a vortex, `vortexing` is enough — don't also tag `mixing`.
