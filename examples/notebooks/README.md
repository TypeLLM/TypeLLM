# Notebooks

Paste your API keys in the cell near the top and run all cells: each notebook calls the APIs itself. card_suit_bets.ipynb, discover_invoice_categories.ipynb, hierarchical_classification.ipynb, time_zones.ipynb and confidence_4_or_9.ipynb keep the outputs of one run, so you can read them before running them.

| Notebook | What it shows | Keys to paste |
| --- | --- | --- |
| [card_suit_bets.ipynb](card_suit_bets.ipynb) | Guess the suit of a random card: TypeLLM and OpenAI's Decisions API give a probability for each suit, then bet against each other. The side closer to the true 1 in 4 wins the pool. | `TYPELLM_API_KEY`, `OPENAI_API_KEY` |
| [confidence_4_or_9.ipynb](confidence_4_or_9.ipynb) | Two nearly identical drawings between a 4 and a 9: TypeLLM, OpenAI's Decisions API, and Cloudflare's Clef and Clef-flash give each a probability and a confidence. Keeps the outputs of one run. | `TYPELLM_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY` |
| [discover_invoice_categories.ipynb](discover_invoice_categories.ipynb) | Sort invoices into categories that start from two: each call picks a category or names a new one. Keeps the outputs of one run. | `TYPELLM_API_KEY` |
| [hierarchical_classification.ipynb](hierarchical_classification.ipynb) | Classify products into a tree of 64 categories: pick a department, then a category inside it, in one call with `when`. Keeps the outputs of one run. | `TYPELLM_API_KEY` |
| [time_zones.ipynb](time_zones.ipynb) | Find each email sender's IANA time zone among all 418, in one choice. Keeps the outputs of one run. | `TYPELLM_API_KEY` |
| [identical_resumes.ipynb](identical_resumes.ipynb) | Two candidates with the same résumé, asked in both orders: the fair answer is 50/50 whichever is listed first. Compares TypeLLM, Jev and OpenAI's Decisions API. | `TYPELLM_API_KEY`, `TYPESAFE_API_KEY`, `OPENAI_API_KEY` |
