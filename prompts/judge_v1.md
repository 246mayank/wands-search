You judge how relevant a product is to a shopper's search query on a furniture and home-goods store.

Give each (query, product) pair one label:

- `Exact`: the product fully matches the query. It is the kind of product asked for, and it satisfies every modifier (material, style, size, colour, feature). If the query names no product type ("shamrock"), a product that clearly has that quality matches.
- `Partial`: the product matches the target entity of the query but doesn't satisfy its modifiers. For "kids chair", a chair that isn't for kids. For "wire basket with dividers", a wire basket without dividers.
- `Irrelevant`: the product is not the kind of thing asked for. For "salon chair", a salon floor mat. For "chair and a half recliner", a massage chair.

Judge only from the text given. Don't penalise a product because a detail is missing from its text unless the text contradicts the query.

Reply with JSON only, in this shape:

{"results": [{"id": 1, "label": "Exact"}]}

Return one result per pair, keeping its `id`.

Pairs:
{{items}}
