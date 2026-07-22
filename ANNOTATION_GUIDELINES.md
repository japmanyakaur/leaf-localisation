Leaf Localisation — Annotation Rules

    One box per visually distinct leaf.
    Compound leaves (strawberry, blueberry, clover, etc.): one box for the whole compound leaf, covering all its leaflets together — not one box per leaflet. (This matches what most of your existing data already does, based on what we checked — so this choice means less rework, not more.)
    Partially visible/occluded leaves: only box a leaf if at least ~20% of it is visible. Skip anything smaller/more hidden than that.
    Box tightness: draw the box tight around just the visible leaf area — no extra background, no stem included unless unavoidable.
    Dense clusters/overlapping leaves: still one box per individual leaf, even if they touch or overlap — don't merge touching leaves into one box just because they're hard to separate.
    