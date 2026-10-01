ARCHITECTURAL_PLAN_PROMPT = r"""
You are analyzing one page of an Indian architectural/building plan.

Your job is semantic interpretation, not blind OCR. Identify drawing regions and
associate visible dimensions with the correct region.

Possible region types include:
SITE_PLAN, GROUND_FLOOR_PLAN, FIRST_FLOOR_PLAN, SECOND_FLOOR_PLAN,
THIRD_FLOOR_PLAN, OTHER_FLOOR_PLAN, AREA_STATEMENT, ELEVATION, SECTION,
ROAD, PARKING, TITLE_BLOCK, SERVICE_DETAIL, OTHER.

For dimensions, use semantic types such as:
PLOT_WIDTH, PLOT_DEPTH, BUILDING_WIDTH, BUILDING_DEPTH, ROAD_WIDTH, BUILDING_HEIGHT,
FLOOR_HEIGHT, PLINTH_HEIGHT, PARAPET_HEIGHT,
FRONT_SETBACK, REAR_SETBACK, LEFT_SETBACK, RIGHT_SETBACK, ROOM_DIMENSION,
PARKING_DIMENSION, BUILDING_HEIGHT, FLOOR_HEIGHT, PLINTH_HEIGHT, PARAPET_HEIGHT, FLOOR_COUNT,
OTHER_DIMENSION, UNKNOWN.

For explicitly labelled areas/coverage/FAR values, use semantic area types:
PLOT_AREA, NET_PLOT_AREA, BUILDING_FOOTPRINT_AREA, PROPOSED_COVERAGE_AREA, PLINTH_AREA,
COVERAGE_PERCENT, FAR_AREA, FAR_RATIO, TOTAL_BUILT_UP_AREA, BUILT_UP_AREA, PLINTH_AREA, UNKNOWN_AREA.

Rules:
1. Do not assume the largest rectangle is the plot.
2. Do not treat room dimensions as plot/building dimensions.
3. Do not calculate a dimension that is not visibly supported.
4. Prefer dimensions whose text and arrows/boundaries are visually associated.
5. A value such as 9.14 and a value such as 8.22 may belong to different regions.
6. Return null when uncertain rather than inventing a value.
7. Bounding boxes use normalized image coordinates [x1,y1,x2,y2] on a 0-1000 grid.
8. Return ONLY valid JSON. No markdown fences.
9. Read each region's dimensions independently, from that region's own pixels only.
   Never reuse or infer a width/depth value from a different region (e.g. a floor
   plan) just because you expect a site plan and a floor plan to match. If the
   same building is drawn rotated between two regions, its printed width/depth
   labels may legitimately swap between the two regions -- report exactly what is
   printed in THIS region, even if it looks inconsistent with another region.
10. FRONT_SETBACK, REAR_SETBACK, LEFT_SETBACK, and RIGHT_SETBACK must only be
    emitted when a numeric label is printed directly on a dimension line drawn
    between the plot boundary and the building boundary on that edge. Do not
    output a "typical" or regulation-looking setback value (e.g. 3.0, 1.5) unless
    you can point to the exact printed digits for it in `evidence`. If no such
    label exists, omit the dimension entirely rather than estimating one.
11. The `evidence` field must be the literal digits/text you read at that bbox,
    not a description of what the dimension represents.
12. Before finishing, inspect the SITE_PLAN corners specifically for small leading-dot
    dimensions such as `.46` and `.47`. These are valid printed dimensions even though
    they do not have a leading zero. Associate each with the corresponding plot/building
    gap and emit FRONT_SETBACK/REAR_SETBACK/LEFT_SETBACK/RIGHT_SETBACK when the side
    can be determined from the road/front orientation.
13. ROAD_WIDTH must be emitted only when a numeric road-width dimension is actually
    printed. The mere presence of a large rectangle labeled ROAD is not enough to
    estimate its width.
12. Setback labels are almost always SMALL decimals (well under 2m on typical
    urban plots) printed right at a corner where the plot boundary and
    building boundary are close together — often as two stacked values at a
    single corner (e.g. ".46" above ".47"). They are visually and numerically
    distinct from PLOT_WIDTH/PLOT_DEPTH/BUILDING_WIDTH/BUILDING_DEPTH, which
    run the full length of an edge. NEVER assign a value you have already
    used for PLOT_WIDTH, PLOT_DEPTH, BUILDING_WIDTH, or BUILDING_DEPTH to a
    setback type too — if you cannot find a distinct, separately-printed
    small value for a given side's setback, omit it.

Return this structure:
{
  "page_number": PAGE_NUMBER,
  "units": null,
  "scale": null,
  "regions": [
    {"id":"region_1","type":"SITE_PLAN","bbox":[0,0,0,0],"confidence":0.0,"label":null,"evidence":null}
  ],
  "dimensions": [
    {"value":0.0,"unit":"m","type":"PLOT_WIDTH","region_id":"region_1","bbox":[0,0,0,0],"evidence":"9.14","confidence":0.0}
  ],
  "areas": [
    {"value":0.0,"unit":"m2","type":"PLOT_AREA","region_id":"region_1","evidence":"222.83","confidence":0.0}
  ],
  "warnings": []
}
"""

SITE_PLAN_FOCUS_PROMPT = r"""
You are given a CROPPED SITE PLAN from an Indian architectural drawing.
This is a focused second-pass Vision extraction. Use only the pixels in this crop.
Do not use knowledge of regulations and do not invent values.

The crop normally contains:
- an outer PLOT/SITE boundary,
- an inner PROPOSED BUILDING boundary,
- dimension labels around the outer boundary,
- small setback labels in the gaps between the two rectangles,
- a ROAD label and its numeric width.

Extract these fields whenever visibly supported:
PLOT_WIDTH, PLOT_DEPTH, BUILDING_WIDTH, BUILDING_DEPTH, ROAD_WIDTH, BUILDING_HEIGHT,
FLOOR_HEIGHT, PLINTH_HEIGHT, PARAPET_HEIGHT,
FRONT_SETBACK, REAR_SETBACK, LEFT_SETBACK, RIGHT_SETBACK.

Also extract explicitly labelled area/coverage/FAR values visible in the crop, using:
PLOT_AREA, NET_PLOT_AREA, BUILDING_FOOTPRINT_AREA, PROPOSED_COVERAGE_AREA, PLINTH_AREA,
COVERAGE_PERCENT, FAR_AREA, FAR_RATIO, TOTAL_BUILT_UP_AREA. Do not calculate an
area from dimensions unless the prompt explicitly marks it as DERIVED in evidence;
prefer the printed area statement when one is visible.

CRITICAL SETBACK RULES:
1. Do NOT require the printed number to say "front", "rear", "left", or "right".
2. Determine the side from spatial position relative to the OUTER plot boundary
   and INNER proposed-building boundary.
3. A number in the gap ABOVE the building is REAR_SETBACK.
4. A number in the gap BELOW the building, adjacent to the road/frontage, is FRONT_SETBACK.
5. A number in the gap LEFT of the building is LEFT_SETBACK.
6. A number in the gap RIGHT of the building is RIGHT_SETBACK.
7. Preserve repeated values. Three separate 0.80 labels are three separate pieces
   of evidence and must not be deduplicated.
8. Do not reuse plot/building dimensions as setbacks.
9. Read small decimals such as 0.80, .80, 1.00 exactly as printed.
10. For BUILDING_WIDTH/DEPTH, use an explicit building dimension if visible in the
    crop. If it is not printed but the inner rectangle is clearly visible and its
    dimensions can be directly derived from the visible plot dimensions and visible
    setback gaps, you may derive it, but set the evidence to the component values and
    confidence lower than a directly printed dimension.
11. ROAD_WIDTH must come from a printed road-width dimension such as "9.20m WIDE ROAD".
12. Never infer a setback from a regulation or from a typical value.
13. Every dimension must include a bbox around the actual printed digits or the exact
    geometry used for a clearly marked derived value.
14. Return ONLY valid JSON.

Return:
{
  "page_number": PAGE_NUMBER,
  "units": "m",
  "scale": null,
  "regions": [
    {"id":"site_focus","type":"SITE_PLAN","bbox":[0,0,1000,1000],"confidence":0.99,"label":"SITE PLAN","evidence":""}
  ],
  "dimensions": [
    {"value":0.0,"unit":"m","type":"PLOT_WIDTH","region_id":"site_focus","bbox":[0,0,0,0],"evidence":"12.19","confidence":0.0}
  ],
  "areas": [
    {"value":0.0,"unit":"m2","type":"PLOT_AREA","region_id":"site_focus","evidence":"222.83","confidence":0.0}
  ],
  "warnings": []
}
"""

AREA_STATEMENT_FOCUS_PROMPT = r"""
You are given a CROPPED AREA STATEMENT TABLE from an Indian architectural drawing.
This is a focused second-pass Vision extraction. Use only the pixels in this crop.
Do not use knowledge of regulations and do not invent values.

The crop normally contains a table with rows such as: Site/Plot Area, one row per
floor (Stilt/Ground/First/.../Terrace) with Gross/Deduction/Nett columns, Total
Built-up Area, Proposed/Ground Coverage Area, Coverage %, and FAR/FSI Achieved
(sometimes as an area in sq.m, sometimes as a bare ratio, sometimes both).

Extract ONLY values that are actually printed as a row/cell in this table, using
these semantic area types:
PLOT_AREA, NET_PLOT_AREA, BUILDING_FOOTPRINT_AREA, PROPOSED_COVERAGE_AREA,
PLINTH_AREA, COVERAGE_PERCENT, FAR_AREA, FAR_RATIO, TOTAL_BUILT_UP_AREA,
BUILT_UP_AREA.

CRITICAL RULES:
1. Read each row's own printed number. Do NOT compute a value from other rows
   (e.g. do not multiply a per-floor area by a floor count, do not divide two
   printed numbers to produce a coverage % or FAR yourself) -- if the table does
   not print a row for a field, omit it entirely rather than deriving it.
2. COVERAGE_PERCENT is a percentage (e.g. "58.11" from "58.11%" or "(58.11%)").
   FAR_RATIO is a bare ratio/decimal (e.g. "1.15"), typically well under 10 --
   never confuse it with FAR_AREA (a number in square metres, typically in the
   hundreds).
3. If the same physical quantity is printed twice in different units (sq.m and
   sq.ft columns), report the sq.m (metric) figure and set unit accordingly;
   do not report both as if they were two different measurements.
4. Every area must include the literal printed text as `evidence` (the row
   label plus the number you read, e.g. "SITE AREA : 160.77 Sq.m").
5. Return null / omit a field entirely when you cannot find its row rather than
   guessing or estimating.
6. Return ONLY valid JSON. No markdown fences.

Return:
{
  "page_number": PAGE_NUMBER,
  "units": "m2",
  "scale": null,
  "regions": [
    {"id":"area_focus","type":"AREA_STATEMENT","bbox":[0,0,1000,1000],"confidence":0.99,"label":"AREA STATEMENT","evidence":""}
  ],
  "dimensions": [],
  "areas": [
    {"value":0.0,"unit":"m2","type":"PLOT_AREA","region_id":"area_focus","evidence":"SITE AREA : 160.77 Sq.m","confidence":0.0}
  ],
  "warnings": []
}
"""

HEIGHT_FOCUS_PROMPT = r"""
You are given a CROPPED ELEVATION or SECTION drawing from an Indian architectural
drawing. This is a focused second-pass Vision extraction. Use only the pixels in
this crop. Do not use knowledge of regulations and do not invent values.

An elevation/section shows the building's vertical dimensions: an overall height
from ground/plinth level to the highest point (parapet or terrace), and often
separate floor-to-floor heights, plinth height, and parapet height as printed
dimension lines running vertically alongside the drawing.

Extract ONLY dimensions that are printed as an explicit numeric label on a
vertical dimension line (or, for the two regulatory/text types below, an
explicit printed sentence) in this crop, using these semantic types:
BUILDING_HEIGHT (ground/plinth level to the highest point of the building --
  if several vertical dimensions are stacked, this is their SUM only when the
  drawing itself prints a separate overall/total figure, or when you can point
  to the single dimension line that spans the full height; otherwise omit it
  rather than adding the stack yourself),
PROPOSED_BUILDING_HEIGHT (the sheet explicitly labels a height figure as the
  "proposed"/"sanctioned" height of THIS building, e.g. "PROPOSED HEIGHT:
  9.60 M" -- functionally the same real-world quantity as BUILDING_HEIGHT,
  kept as a distinct type only because the sheet itself used a distinct
  label; the deterministic layer treats the two identically as the
  building's actual height, never as a regulatory ceiling),
FLOOR_HEIGHT (one floor-to-floor dimension, e.g. "3.00" between two floor
  slabs -- if several floors repeat the same value, report it once),
STILT_HEIGHT (floor-to-floor height of a stilt/parking level specifically --
  report this instead of FLOOR_HEIGHT when the drawing itself labels that
  level "STILT"; do not assume every ground level is a stilt),
PLINTH_HEIGHT (ground level to plinth/floor-one level),
PARAPET_HEIGHT (topmost floor/terrace level to the top of the parapet wall),
HEIGHT_EXCLUDING_STILT (the sheet explicitly prints a height figure labeled
  as excluding the stilt level, e.g. "HEIGHT EXCL. STILT: 12.50 M" -- this is
  a DIFFERENT quantity from BUILDING_HEIGHT and must never be merged with
  it; report it only when the sheet's own label says "excluding stilt"),
REGULATORY_MAX_HEIGHT (a maximum/permitted height ceiling from a regulation
  citation printed on the sheet, e.g. "MAX HEIGHT PERMISSIBLE AS PER RULE:
  15.0 M" -- a legal ceiling, never this building's actual/proposed height),
REGULATORY_TEXT (any other regulatory height text that doesn't fit
  REGULATORY_MAX_HEIGHT above but is still clearly a rule citation, not a
  measurement of this building).

CRITICAL RULES:
1. Do NOT sum floor heights yourself to invent BUILDING_HEIGHT -- if the sheet
   does not print an explicit overall height figure or a single dimension line
   spanning the full height, leave BUILDING_HEIGHT out; the deterministic layer
   downstream derives it from FLOOR_HEIGHT x floor_count when appropriate, and
   must not receive a value that already silently did that.
2. NEVER report a maximum/permitted/regulatory height figure as BUILDING_HEIGHT
   or PROPOSED_BUILDING_HEIGHT. A regulatory ceiling and this building's own
   actual height are different numbers even when they happen to be equal --
   classify by what the sheet's own text says the number IS (a rule citation
   vs. a measurement of this specific building), never by which type would be
   more useful downstream.
3. A height value must include the literal printed digits (or, for
   REGULATORY_MAX_HEIGHT/REGULATORY_TEXT, the literal printed sentence) as
   `evidence`.
4. Return null / omit a field entirely when no explicit printed dimension line
   or sentence supports it, rather than estimating one from the drawing's
   proportions.
5. Return ONLY valid JSON. No markdown fences.

Return:
{
  "page_number": PAGE_NUMBER,
  "units": "m",
  "scale": null,
  "regions": [
    {"id":"height_focus","type":"ELEVATION","bbox":[0,0,1000,1000],"confidence":0.99,"label":"ELEVATION","evidence":""}
  ],
  "dimensions": [
    {"value":0.0,"unit":"m","type":"BUILDING_HEIGHT","region_id":"height_focus","bbox":[0,0,0,0],"evidence":"9.60","confidence":0.0}
  ],
  "areas": [],
  "warnings": []
}
"""

DXF_VISION_FALLBACK_PROMPT = r"""
You are looking at a rendered line drawing extracted from a DXF architectural
site plan. It shows the raw geometry as black lines/outlines on a white
background -- there may be NO text, NO layer colors, and NO dimension labels
visible, because they were not present in the source file. Some lines may be
noisy/fragmented (many short disconnected segments forming what should be one
edge, e.g. a dashed boundary that got traced as many tiny marks).

Your job is purely SPATIAL/SEMANTIC: identify which region of the drawing is
which kind of object, using shape and position only -- you are NOT being
asked to read or estimate any measurement, scale, or number.

Identify, if visually identifiable:
- SITE_PLAN: the overall plot/site boundary (usually the largest closed or
  near-closed outline enclosing everything else).
- BUILDING: the building footprint -- typically a smaller, roughly
  rectilinear closed or near-closed shape nested inside the site boundary,
  usually showing internal wall/room subdivisions if any detail is visible.
  Do NOT pick an isolated tiny mark, a single stray line, or scattered noise
  -- the building is a coherent, recognizably building-shaped region.
- ROAD: a rectangular strip-shaped region adjacent to (touching or just
  outside) the site boundary on one side, if visible.

Rules:
1. Return a bounding box for a category ONLY if you can actually see a
   coherent shape matching that description. If nothing plausible is
   visible for a category, omit it entirely -- do not guess or default to
   the whole image.
2. Bounding boxes use normalized image coordinates [x1,y1,x2,y2] on a
   0-1000 grid, tightly fitted to the identified shape (not the whole image
   unless the shape genuinely fills it).
3. Do not report any "value", "areas", or "dimensions" -- only "regions".
   No numeric measurement of any kind belongs in your answer; the exact
   measurement is computed separately, directly from the source vector data
   within whichever region you identify.
4. Return ONLY valid JSON. No markdown fences.

Return:
{
  "page_number": 1,
  "units": null,
  "scale": null,
  "regions": [
    {"id":"region_1","type":"SITE_PLAN","bbox":[0,0,1000,1000],"confidence":0.0,"label":null,"evidence":null},
    {"id":"region_2","type":"BUILDING","bbox":[0,0,0,0],"confidence":0.0,"label":null,"evidence":null}
  ],
  "dimensions": [],
  "areas": [],
  "warnings": []
}
"""
