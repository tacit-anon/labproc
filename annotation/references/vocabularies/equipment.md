# Equipment vocabulary — `equipment_tags` column

Multi-select. Apply tags for all apparatus visibly present in the frame, including supporting equipment in the background. Comma-separated.

## Containers / glassware

| tag | description |
|---|---|
| `erlenmeyer_flask` | cone-shaped flat-bottomed flask |
| `round_bottom_flask` | spherical flask, often with ground-glass joint |
| `beaker` | straight-sided cylinder with pour spout |
| `graduated_cylinder` | tall thin volumetric cylinder |
| `sep_funnel` | pear-shaped separating funnel with stopcock |
| `dropping_funnel` | similar to sep funnel but for controlled addition |
| `volumetric_flask` | pear-shaped with neck and calibration mark |
| `buchner_funnel` | porcelain funnel with perforated plate |
| `fluted_funnel` | glass funnel with fluted (pleated) filter paper |
| `glass_funnel` | plain glass funnel, no filter |
| `powder_funnel` | wide-stem funnel for solids |
| `test_tube` | small cylindrical glass tube |
| `microtube` | 0.5/1.5/2 mL plastic Eppendorf-style tube |
| `pcr_tube` | 0.2 mL PCR tube (often colored caps) |
| `vial` | small glass vial with cap |
| `petri_dish` | flat round dish |
| `column_chromatography` | vertical glass column, often with stopcock at base |
| `tlc_plate` | flat silica-coated TLC plate |
| `tlc_chamber` | jar/chamber for TLC development |

## Heating equipment

| tag | description |
|---|---|
| `hot_plate` | flat heating plate (often combined stirrer) |
| `heating_mantle` | bowl-shaped heater for round-bottom flasks |
| `bunsen_burner` | open flame |
| `oil_bath` | flask immersed in oil |
| `water_bath` | flask immersed in water |
| `dry_bath` | metal block heater |
| `microwave` | lab microwave (e.g., for agarose) |
| `oven` | drying oven |

## Cooling / condensation

| tag | description |
|---|---|
| `ice_bath` | beaker / dewar with ice |
| `condenser_vertical` | reflux condenser pointing up |
| `condenser_horizontal` | distillation condenser at angle |
| `freezer` | freezer / cold room |

## Mixing / agitation

| tag | description |
|---|---|
| `magnetic_stirrer` | stirrer plate with stir bar |
| `vortex_mixer` | vortex mixer (Genie, etc.) |
| `orbital_shaker` | rotating platform shaker |
| `rocker` | side-to-side rocker |

## Measurement / analysis

| tag | description |
|---|---|
| `analytical_balance` | enclosed-pan precision balance |
| `top_loading_balance` | open-pan general balance |
| `pipette` | air-displacement pipette (single channel) |
| `multichannel_pipette` | multichannel pipette |
| `serological_pipette` | glass/plastic graduated pipette + electronic dispenser |
| `burette` | graduated tube with stopcock for titration |
| `thermometer` | thermometer |
| `ph_meter` | pH meter / probe |

## Pumps / vacuum

| tag | description |
|---|---|
| `vacuum_pump` | vacuum line / pump for filtration or distillation |
| `peristaltic_pump` | peristaltic pump for fluids |

## Specialized apparatus

| tag | description |
|---|---|
| `rotovap` | rotary evaporator |
| `distillation_apparatus` | full distillation rig (flask + condenser + receiver) |
| `transfer_apparatus` | Western blot transfer cassette / tank |
| `electrophoresis_tank` | gel running tank with electrodes |
| `gel_casting_tray` | tray + comb for casting gels |
| `thermocycler` | PCR thermocycler / isothermal device |
| `microcentrifuge` | mini benchtop centrifuge |
| `extraction_column` | silica spin column for nucleic acid prep |

## Lab infrastructure

| tag | description |
|---|---|
| `ring_stand` | metal rod stand with clamps |
| `fume_hood` | fume hood enclosure |
| `biosafety_cabinet` | BSC for cell culture |
| `bench_top` | open lab bench |

## Rules

- Apply tags only for what's clearly visible. If a hot plate is partially obscured, skip it.
- Don't tag generic context (lab coat, gloves) unless they're informative.
- Multiple containers in one frame: tag all relevant ones (e.g., a transfer scene might have `erlenmeyer_flask, fluted_funnel, ring_stand`).
- For specialized apparatus, prefer the specific tag (`rotovap`) over generic component tags (`condenser_vertical`).
