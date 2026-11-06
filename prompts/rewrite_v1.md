You clean up search queries for a furniture and home-goods store, so a catalogue search can find what the shopper meant.

For each query:

- Fix spelling and expand abbreviations ("pedistole sink" becomes "pedestal sink").
- Remove words that don't describe a product.
- Keep every detail the query gives. Keep brand, collection and model names exactly as written.
- Don't add details the query doesn't have.
- If the query is already fine, or you can't tell what it means, return it unchanged.

Reply with JSON only, in this shape:

{"results": [{"id": 1, "rewrite": "..."}]}

Return one result per query, keeping its `id`.

Queries:
{{items}}
