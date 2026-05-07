# Label catalog

These are the canonical labels per category. The `value` column is what goes into the spreadsheet's `your_label` and `physical_state` columns. The `description` is what the frame must visibly demonstrate.

## Organic Purification (`branch=op`, `category=organic_purification`)

| value | description |
|---|---|
| `mixture_crude_unreacted` | raw reaction mixture (cloudy/impure starting material, before heating) |
| `mixture_dissolved_hot` | dissolved in hot solvent (clearer liquid, in stoppered flask, being transferred, or being filtered hot) |
| `crystals_forming` | crystals beginning to appear / visible in solution but not yet collected |
| `crystals_complete` | fully crystallized — isolated and collected (typically on Buchner filter or as dry product) |
| `tlc_plate_dry` | TLC plate before spotting |
| `tlc_plate_spotted` | spots applied to plate |
| `tlc_running` | plate in developing chamber |
| `tlc_developed` | developed plate with visible spots |
| `column_dry` | empty or dry column |
| `column_packed` | column packed with silica/media |
| `column_equilibrated` | solvent running through, equilibrating |
| `sample_loaded` | sample applied to top of column |
| `fractions_collecting` | fractions being collected |
| `fraction_analysis` | analyzing collected fractions |
| `lle_two_phase_settled` | separating funnel with two phases visibly separated (post-mixing, pre-draining) |
| `lle_draining_lower_layer` | draining the lower phase out of the separating funnel into a receiving vessel |
| `distillation_setup_running` | heated distilling flask + condenser active, vapor moving through the apparatus |
| `distillate_collecting` | distillate accumulating in the receiving vessel (graduated cylinder, flask, or vial) |
| `reflux_running` | flask under a vertical condenser, solvent visibly boiling and condensing back into the flask |
| `rotovap_running` | rotary evaporator with bath, flask actively spinning under vacuum, concentrating solvent |
| `vacuum_filtration_general` | Buchner funnel + filter flask under vacuum, collecting solid from any solution (not strictly crystallization) |
| `analytical_weighing` | enclosed-pan analytical balance with a sample on it, weight readout visible (typically 4-decimal precision) |
| `solvent_dispensing` | solvent being measured / poured from a reagent bottle into a vessel, often in a fume hood with hazard-labeled bottles |
| `titration_running` | burette dripping into a stirred sample (magnetic stirrer + beaker), indicator color often visible in the burette or sample |
| `gravity_filtration_hot` | hot solution being filtered through a fluted filter paper in a glass funnel into a receiving flask (distinct from vacuum filtration) |

## PCR (`branch=pcr`, `category=pcr`)

| value | description |
|---|---|
| `dry_tube_no_liquid` | tube is empty or dry |
| `liquid_added_unmixed` | liquid visible but not mixed |
| `liquid_mixed_clear` | clear mixed liquid |
| `liquid_viscous` | thick/viscous liquid visible |
| `gel_unset` | gel solution poured but not set |
| `gel_set_bands_loading` | solid gel, samples being loaded |
| `gel_running` | gel in tank with current running |
| `gel_complete_bands_visible` | finished gel with visible bands |
| `tube_in_vortex` | microtube held against a vortex mixer, sample being mixed |
| `tube_in_microcentrifuge` | tubes spinning in microcentrifuge or mini-centrifuge (apparatus visible) |
| `tube_in_thermocycler` | PCR tubes loaded into thermocycler or isothermal heater, instrument running |
| `dry_bath_incubating` | tubes incubating in dry bath / heat block at fixed temperature (no spinning) |
| `extraction_column_loading` | sample being pipetted onto a silica spin column for nucleic acid extraction |
| `agarose_dissolving` | agarose being microwaved / dissolved in buffer (pre-pour stage, before `gel_unset`) |

## Western Blot (`branch=wb`, `category=western_blot`)

| value | description |
|---|---|
| `gel_uncast` | no gel yet, casting setup visible |
| `gel_set_dry` | solid gel, not yet in buffer |
| `gel_running` | gel in tank, electrophoresis running |
| `gel_complete` | gel finished, removed from tank |
| `membrane_dry` | membrane visible, not yet wet |
| `membrane_wet_no_transfer` | membrane in liquid, transfer not started |
| `membrane_transfer_complete` | transfer done, membrane has protein |
| `membrane_blocking` | membrane in blocking solution (milk/BSA) |
| `membrane_primary_antibody` | membrane in antibody solution |
| `membrane_washing` | membrane being washed |
| `membrane_secondary_antibody` | second antibody incubation |
| `membrane_ecl_exposure` | detection/exposure step |
| `bands_visible` | final result, protein bands visible |
| `gel_staining` | gel submerged in stain or destain solution, blue/yellow color visible (often on a rocker tray) |
| `comb_removed` | gel comb being lifted from a set gel, wells now exposed |
| `transfer_setup_assembling` | transfer cassette/sandwich being prepared (filter-membrane-gel-filter) or being placed in transfer tank with ice |
| `protein_sample_with_buffer` | protein sample being mixed with loading/sample buffer (e.g., Laemmli) prior to loading into gel |
| `gel_apparatus_cleaning` | gel cassette / electrophoresis equipment being rinsed in sink after a run |
| `buffer_preparation` | running/transfer/TBS buffer being prepared in a graduated cylinder or flask |

## Branch ↔ category mapping

| friendly category | branch code |
|---|---|
| `organic_purification` | `op` |
| `pcr` | `pcr` |
| `western_blot` | `wb` |
