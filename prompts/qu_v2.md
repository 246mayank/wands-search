You analyse furniture and home-goods search queries for an online store's catalogue.

For each query below, return:

- `product_classes`: up to 3 classes, best first, for the kind of product the shopper wants. Each must be copied exactly from that query's `candidates`. Put the class a store would file the product under first, and prefer a specific class over a general one. Use an empty list only if nothing fits.
- `attributes`: only what the query itself states. Allowed keys: color, material, style, shape, pattern, room, size, theme. Leave out a key the query doesn't state. Don't guess.
  - A brand, collection or model name is not an attribute, even if it sounds like a style. Leave it out.
  - Write the value a catalogue would use: "blk" is "black", and "living" is "living room".
  - A dimension or a count ("48 inch", "4 light") is not an attribute. Leave it out. "king", "queen" and "twin" are sizes.
- `intent`: one of `specific_item` (a named product or model), `category_browse` (a kind of product), `style_or_room` (a look or a room rather than one product type), `unclear`.
- `confidence`: `high`, `medium` or `low`, for how sure you are about `product_classes`.

Reply with JSON only, in this shape:

{"results": [{"id": 1, "product_classes": ["..."], "attributes": {"color": "..."}, "intent": "...", "confidence": "..."}]}

Return one result per query, keeping its `id`.

Queries:
{{items}}
