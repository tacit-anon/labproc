# Substance vocabulary — `substance_tags` column

Multi-select. Apply all tags that visibly describe the substance(s) inside the primary vessel(s) in the frame. Comma-separated in the spreadsheet.

## Phase / state

| tag | description |
|---|---|
| `liquid` | freely flowing liquid |
| `solid` | solid material (chunk, crystal, powder, residue) |
| `gas_vapor` | visible vapor / steam / smoke |
| `multi_phase` | two or more phases visibly distinct (e.g., LLE) |
| `suspension` | particulate solid suspended in liquid |
| `slurry` | dense crystal/precipitate suspension just before filtration |
| `foam` | bubble/foam layer |
| `powder` | dry powder |
| `crystalline` | clearly crystalline solid (faceted / shiny solid mass) |

## Condition

| tag | description |
|---|---|
| `hot` | actively heated (hot plate on, steam, condensation) — visible cue required |
| `cool` | room temp or cooled (ice bath, no heat) |
| `frozen` | solidified by cold / ice |
| `dissolved` | solid fully dissolved in solvent (clarity is the cue) |
| `undissolved` | solid still visible alongside solvent |
| `settled` | phases or particles have visibly separated and stopped moving |
| `stirred` | actively mixing (magnetic stirrer or by hand) |
| `boiling` | active boiling — bubbles rising |
| `condensing` | condensation visible on glass surfaces |

## Opacity

| tag | description |
|---|---|
| `clear` | fully transparent, can read text through |
| `translucent` | light passes but features blurred |
| `cloudy` | light passes but visibly cloudy / hazy |
| `opaque` | no light passes |

## Color (apply one or more)

| tag | description |
|---|---|
| `colorless` | water-clear, no tint |
| `white` | white / pale / milky |
| `yellow` | yellow / pale yellow |
| `orange` | orange |
| `red` | red / pink / dark red |
| `brown` | brown / amber / tan |
| `green` | green |
| `blue` | blue / cyan |
| `purple` | purple / violet / magenta |
| `black` | black or very dark |

## Examples

- A flask of clear hot solution being heated: `liquid, hot, dissolved, clear, colorless`
- A sep funnel after shaking, two layers visible (red over yellow): `liquid, multi_phase, settled, red, yellow`
- White crystals on a Buchner filter: `solid, crystalline, cool, opaque, white`
- A rotovap concentrating brown extract: `liquid, hot, opaque, brown` (add `condensing` if visible)

## Rules

- **Apply at least one phase tag** (`liquid`, `solid`, `gas_vapor`, `multi_phase`, `suspension`, `slurry`, `foam`, `powder`, `crystalline`).
- **Apply at least one opacity tag** when liquid is visible.
- **Apply at least one color tag** when color is determinable. Skip if colorless and unambiguous; else use `colorless` explicitly.
- Don't pile on speculative tags. Only what's visibly evident in this frame.
