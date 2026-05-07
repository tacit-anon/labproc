# Disambiguation rules

These rules resolve ambiguity that pure visual inspection can miss. Apply them before assigning labels — most labeling errors come from skipping these, not from misseeing the frame.

## Anti-anchoring: the equipment-scan shortcut

Before reaching for any specific label, do an equipment scan on the frame and eliminate whole label groups by what is NOT present. For OP, this dramatically narrows the search space:

| equipment NOT in frame | labels eliminated |
|---|---|
| no TLC plate | `tlc_plate_dry`, `tlc_plate_spotted`, `tlc_running`, `tlc_developed` |
| no chromatography column | `column_dry`, `column_packed`, `column_equilibrated`, `sample_loaded`, `fractions_collecting`, `fraction_analysis` |

In a typical recrystallization video, both eliminations apply — leaving only `mixture_crude_unreacted`, `mixture_dissolved_hot`, `crystals_forming`, `crystals_complete` as candidates. State this elimination ONCE per video in your audit log; apply silently per frame.

Same logic for PCR (gel apparatus vs liquid in tube) and WB (gel vs membrane stages).

The point of the equipment scan is not to be lazy — it's to keep the choice space honest. A label picked from "all 14 candidates" is fundamentally different from a label picked from "the only 4 still possible after eliminating apparatus that isn't there." Be explicit about which one you're doing.



## OP — Crystals: forming vs complete

- `crystals_forming`: crystals are visible **in solution** — in a flask, in a slurry, in a cooling/ice bath, being prepared for filtration. Even a dense, milky-white crystal mass that hasn't been physically separated from the mother liquor is still `forming`.
- `crystals_complete`: crystals are **isolated and collected** — typically resting on a Buchner funnel filter under vacuum, or shown as a dry product. The defining trait is **physical separation from the liquid**, not visual density.

If you're unsure, ask: "Has filtration happened yet in this video?" If no, label is `forming`. If yes and the frame shows the post-filter crystals, label is `complete`.

## OP — Crude vs dissolved

- `mixture_crude_unreacted`: cloudy, opaque, or impure-looking mixture. The "raw reaction mixture" before any heating/dissolving step. Often shown in a flask with a stopper, looking milky or with visible particulate.
- `mixture_dissolved_hot`: mixture has been dissolved in hot solvent. Liquid will look clearer (often slightly colored — pale yellow is common) but mostly transparent. Often shown:
  - In a stoppered flask post-heating
  - Being poured between flasks (transfer step)
  - Being filtered through fluted filter paper (hot gravity filtration)
  - Sitting after activated carbon (decolorization) treatment

The transition from crude to dissolved is the heating step. If you can't tell which side of the transition you're on, default to `mixture_dissolved_hot` at `low` confidence.

## OP — Procedural ordering

Standard recrystallization sequence:
1. `mixture_crude_unreacted` (impure starting material)
2. `mixture_dissolved_hot` (heat to dissolve)
3. *Optional decolorization* — add activated carbon, stir hot — no dedicated label, use `mixture_dissolved_hot` at `low`
4. *Hot gravity filtration* through fluted filter paper — no dedicated label, use `mixture_dissolved_hot` at `medium`
5. Cool the filtrate (often ice bath) → `crystals_forming` as crystals nucleate
6. *Vacuum filtration* through Buchner funnel
7. `crystals_complete` once crystals are isolated on the filter
8. Optional: `tlc_*` family for purity check, `fraction_analysis` for column-purified products

If the video shows step N, earlier-step labels are still possible (someone might re-show an earlier flask). Later-step labels are NOT possible.

## OP — Liquid-liquid extraction (LLE)

- `lle_two_phase_settled`: sep funnel held vertical, two distinct liquid phases visible (often colored — red over yellow, brown over clear, etc.). User has shaken/mixed and is now waiting for separation, OR is showing the settled state before draining.
- `lle_draining_lower_layer`: stopcock open, lower phase flowing out into a receiving vessel (beaker, flask, or another sep funnel). Hand typically on the stopcock.

LLE without crystallization context is its own thing — don't try to fit it under `mixture_*`.

## OP — Distillation (steam, simple, fractional)

- `distillation_setup_running`: distilling flask on heat (mantle, hot plate, or open flame) with vapor visibly rising through a connected condenser. Often shown with the full apparatus in frame.
- `distillate_collecting`: focus on the receiving vessel — drops accumulating from the condenser tip into a graduated cylinder, flask, or vial. The act of collection.

These two often coexist in the same frame; pick the one whose subject is more prominent. If the wide apparatus shot is the subject, use `distillation_setup_running`. If the camera is zoomed on the drip point, use `distillate_collecting`.

Steam distillation specifically: plant material being loaded into the distilling flask before the run starts has no dedicated label — use `mixture_crude_unreacted` only if the user explicitly accepts plant material as "crude," otherwise skip.

## OP — Reflux

- `reflux_running`: flask under a vertical condenser, solvent boiling and condensing back into the flask. NO collection vessel below the condenser (that distinguishes it from distillation). Common in crude product workup or reaction setup before purification.

## OP — Concentration / drying

- `rotovap_running`: rotary evaporator clearly visible — bath (water/oil), motor unit, flask attached and spinning, condenser receiving solvent. The "rotovap" form factor is unmistakable.

## OP — General vacuum filtration

- `vacuum_filtration_general`: Buchner funnel + filter flask + vacuum line, collecting any solid from any solution. Distinct from `crystals_complete` because the solid being collected isn't necessarily a crystallization product (could be a precipitate, a powder, a dye filtrate test).

When the substance IS clearly crystals from a recrystallization workflow, prefer `crystals_complete`. Use `vacuum_filtration_general` for the broader case (dye filtration demos, generic precipitate collection).

## OP — Analytical and prep steps

The original taxonomy was crystallization + chromatography focused. The following labels cover broader synthesis-and-analysis workflows that show up in the corpus:

- `analytical_weighing`: enclosed-pan analytical balance (Mettler, Fujitsu, Sartorius, etc.) with a sample on the pan and a digital readout visible. Distinct from a generic top-loading balance — the glass enclosure and 4-decimal precision are the cues. Common in synthesis prep and gravimetric analysis.
- `solvent_dispensing`: solvent (n-Hexane, ethanol, acetic acid, methanol, etc.) being measured or poured from a brand-labeled reagent bottle into a flask, beaker, or measuring cylinder. Usually in a fume hood. Hazard-labeled brown glass bottles are the signature.
- `titration_running`: burette mounted on a stand, drops falling into a beaker on a magnetic stirrer below. Indicator color (red/pink/blue depending on indicator + endpoint) often visible. Distinct from `lle_draining_lower_layer` — burette is *thin and graduated*, sep funnel is *pear-shaped with a stopper*.
- `gravity_filtration_hot`: a fluted (pleated) filter paper sitting in a glass funnel, hot solution being poured through, filtrate collecting in an Erlenmeyer below. **Use this label rather than `mixture_dissolved_hot`** for filtration frames — `mixture_dissolved_hot` is a substance state, `gravity_filtration_hot` is the operation. They can coexist if both are clearly visible.

## OP — Filtration steps without dedicated labels

The OP label set lacks explicit labels for:
- gravity hot filtration
- vacuum filtration
- decolorization (activated carbon)

When the visible state is hot solution being filtered or decolorized, label as `mixture_dissolved_hot` (the substance state) at `low` or `medium` confidence — depending on how clearly the dissolved liquid is visible. Don't try to force-fit a chromatography label (`column_*`) or a TLC label.

When the action is collecting crystals onto a Buchner filter and the crystals are clearly isolated from the liquid, label as `crystals_complete`.

## PCR — Sample prep & instrument operations

The PCR workflow has three upstream phases the original 8 labels didn't cover. Use these for the corresponding visual cues:

- `tube_in_vortex`: vortex mixer apparatus visible, tube pressed against the head, sample being mixed by vortex action. Distinct from `liquid_added_unmixed` which is the *state* of unmixed liquid in a tube — `tube_in_vortex` is the *action* of mixing.
- `tube_in_microcentrifuge`: microcentrifuge / mini-centrifuge with rotor visible, tubes loaded, lid closed and spinning (or rotor visibly engaged). The "spinning down" action.
- `tube_in_thermocycler`: PCR tubes loaded in a thermocycler block or isothermal heater (e.g., MastiSensor, Genie 2). Distinct from `dry_bath_incubating` because the thermocycler runs a temperature *program* (multiple steps); a dry bath holds at one temperature.
- `dry_bath_incubating`: tubes in a heat block / dry bath at a single temperature (often shown with temperature display reading 37°C, 70°C, etc.). No spinning, no PCR program.
- `extraction_column_loading`: sample being pipetted onto a silica spin column (Qiagen-style or similar). Often shown with multi-well column racks. The column has a colored cap/frit visible. This is the loading step before centrifugation.
- `agarose_dissolving`: agarose powder being dissolved in TAE/TBE buffer — typically a flask of cloudy → clear liquid being microwaved or just removed from the microwave (often foil-wrapped, hot). This precedes `gel_unset` (which is post-pour, pre-solidification).

## PCR — Liquid states

- `dry_tube_no_liquid`: tube empty, no liquid visible
- `liquid_added_unmixed`: liquid visible but droplets/layers not yet mixed (often after pipetting before vortex)
- `liquid_mixed_clear`: homogeneous clear mixed liquid
- `liquid_viscous`: thick, gel-like, or distinctly viscous liquid

## PCR — Gel transitions

- `gel_unset`: liquid agarose freshly poured, comb visible, not yet solidified
- `gel_set_bands_loading`: solid gel, comb removed, samples being pipetted into wells
- `gel_running`: gel in electrophoresis tank with leads attached, current running (look for buffer covering the gel and tank)
- `gel_complete_bands_visible`: gel out of tank or stained, bands visible under UV/visible light

## Western Blot — Casting & gel handling extensions

The original WB labels jump from `gel_set_dry` directly to `gel_running`, but real WB workflows have several intermediate physical states worth labeling:

- `gel_uncast`: glass plates / casting frame assembled, no gel poured yet (water or buffer being added between plates is a common preparation cue).
- `comb_removed`: comb being lifted out of a set gel, exposing the loaded wells. Distinct from `gel_set_bands_loading` (which is the *act* of loading samples into wells).
- `gel_staining`: post-electrophoresis gel sitting in a colored stain or destain solution — Coomassie blue is the classic case; yellow/clear destained gels also fit. Often on an orbital shaker or tray.
- `gel_apparatus_cleaning`: gel cassette, glass plates, or tank being rinsed in a sink after a run. End-of-procedure cleanup.

## Western Blot — Sample & buffer preparation

- `protein_sample_with_buffer`: protein lysate being mixed with sample/loading buffer (Laemmli, etc.) in a microtube before loading. Typically shown with pipetting into a small tube, gel apparatus visible in the background or workspace context.
- `buffer_preparation`: a buffer being prepared from concentrate or scratch in a graduated cylinder, beaker, or flask. Running buffer (Tris-Glycine-SDS), transfer buffer, TBS, blocking buffer, etc. The container is the cue (volumetric vessel, no gel/membrane in frame).

## Western Blot — Transfer setup

- `transfer_setup_assembling`: the transfer "sandwich" being built (foam pad, filter paper, gel, membrane, filter paper, foam pad) and/or placed into a transfer tank with ice. Distinct from `membrane_wet_no_transfer` (which is just the wet membrane state) — `transfer_setup_assembling` is the active assembly step.

## Cross-category labels

Some labels apply across categories — don't duplicate:

- `tube_in_vortex` (defined under PCR): also applies to WB sample-prep frames showing a tube being vortexed against a vortex mixer. Use the PCR label as-is.
- `tube_in_microcentrifuge` (PCR): applies to WB lysate clarification spins.
- `dry_bath_incubating` (PCR): applies to WB sample heat-denaturation (95°C boil with Laemmli buffer).

When a frame from a WB video shows a procedure that has a clear PCR or OP label, use that label rather than forcing a WB-specific name. The `branch` column always reflects the source category, but labels are reusable.

## Western Blot — Membrane treatment sequence

Follows the WB protocol order strictly:
1. `gel_uncast` → `gel_set_dry` → `gel_running` → `gel_complete`
2. Then transfer setup: `membrane_dry` → `membrane_wet_no_transfer` → `membrane_transfer_complete`
3. Then antibody workflow: `membrane_blocking` → `membrane_primary_antibody` → `membrane_washing` → `membrane_secondary_antibody` → `membrane_washing` (again, optional)
4. Then detection: `membrane_ecl_exposure` → `bands_visible`

If a frame shows two stages (e.g., transitioning from blocking to primary antibody), pick the most prominent state in the frame.

## When to skip (universal)

Skip a frame entirely — don't label it — when:
- It's a title card, intro screen, or text-only slide
- The apparatus is assembled but no procedural action is visible (pure setup shot)
- The frame is visually identical to an immediately adjacent frame (redundant)
- Multiple labels would each fit equally — the ambiguity is irreducible
- The frame is cut, blurry, or out-of-focus to the point of unreadability

A skip is more valuable than a wrong label. The downstream user reviews skipped frames manually if needed.

## Confidence calibration

- `high`: I can articulate exactly which descriptor in the label catalog this frame demonstrates, and no competing label fits.
- `medium`: The label is the best fit but a competing label is plausible, or the visual cue is partial.
- `low`: The label is a procedural inference (e.g., labeling a filtration action under a substance-state label) rather than a direct visual match.

If you're tempted to set confidence below `low`, skip instead.
