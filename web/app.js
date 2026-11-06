const form = document.getElementById("search-form");
const input = document.getElementById("query");
const results = document.getElementById("results");
const understanding = document.getElementById("understanding");

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function describeUnderstanding(qu) {
  if (qu.status !== "OK") return `Query understanding: none (${qu.status}), so no boost applied.`;
  const attributes = Object.entries(qu.attributes).map(([key, value]) => `${key}: ${value}`);
  const parts = [`classes: ${qu.product_classes.join(", ") || "none"}`, `intent: ${qu.intent}`];
  if (attributes.length) parts.push(attributes.join(", "));
  return `Query understanding: ${parts.join(" · ")}`;
}

function whyBlock(why) {
  const details = element("details");
  details.append(element("summary", "Why this result"));
  const list = element("dl");
  for (const [key, value] of Object.entries(why)) {
    list.append(element("dt", key));
    list.append(element("dd", typeof value === "number" ? String(Math.round(value * 1000) / 1000) : value));
  }
  details.append(list);
  return details;
}

function card(result) {
  const article = element("article", undefined, "card");
  article.append(element("h2", result.name));
  const rating = result.rating === null ? "no rating" : `${result.rating} stars (${result.rating_count})`;
  article.append(element("p", `${result.class || "no class"} · ${rating}`, "meta"));
  article.append(whyBlock(result.why));
  return article;
}

async function search(query) {
  results.replaceChildren(element("p", "Searching..."));
  const response = await fetch(`/search?q=${encodeURIComponent(query)}&k=24`);
  if (!response.ok) {
    results.replaceChildren(element("p", `Search failed (${response.status}).`));
    return;
  }
  const body = await response.json();
  understanding.textContent = describeUnderstanding(body.understanding);
  results.replaceChildren(...body.results.map(card));
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  search(input.value.trim());
});
