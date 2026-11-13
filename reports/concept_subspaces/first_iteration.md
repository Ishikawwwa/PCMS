# Per-concept memory subspaces — first iteration

## Question and setup

Can a confirmed purchase or travel constraint be recovered through another memory's table reference? The prototype stores each annotated memory once, indexes it under multiple concepts, keeps source links, and resolves current table rows at an evidence cutoff. For shopping, the prior slots must all have confirmed prices before the system computes remaining budget. For travel, the latest confirmed row at the cutoff supplies the applicable value.

Task 24 uses the three purchase summaries in the [MAGE paper, Case Study A](https://arxiv.org/pdf/2606.06090v1#page=15). The Mermaid distractor comes from the SimpleMem excerpt and is injected with an assigned position. The travel fixtures use the lab brief's worked example plus synthetic confirmation and scope checks. Concept IDs, evidence status, relevant rows, and the $100 bundle budget are supplied as fixture annotations. The slot-resolved case ranks memories from already resolved concept IDs. The missed-predecessor and travel cases use fixed initial recall. Jev, semantic routing, raw-observation extraction, and actor behavior are unmeasured.

## Results

Sixty offline reads cover five scenarios, four methods, and three memory-token ceilings. The original expansion ablation holds the selected initial memories, packing order, and ceiling fixed. A second ablation gives decision evidence first priority using the same initial selection and ceiling. Direct table lookup receives the same annotated scope and ceiling. It has no initial concept recall.

| Scenario at 96 tokens | Concept recall | Recall then expand | Expand with decision evidence first | Direct table |
| --- | --- | --- | --- | --- |
| Shopping, predecessor slot resolved | Predecessor only | Predecessor only | Complete, 79 tokens | Complete, 79 tokens |
| Shopping, predecessor missed | Incomplete | Incomplete | Complete, 79 tokens | Complete, 79 tokens |
| Shopping, only considered item recalled | Incomplete | Incomplete | Incomplete | Complete, 79 tokens |
| Travel, increase proposed | Incomplete | $80 limit recovered | $80 limit recovered | $80 limit recovered |
| Travel, increase confirmed | Incomplete | $100 limit recovered | $100 limit recovered | $100 limit recovered |

At 192 tokens, recall then expand restores Product 3 and the $34.74 balance in the missed-predecessor case using 165 tokens. At 96 tokens, the same packing fills 85 tokens with frosting and the considered cake cup, leaving insufficient room for the predecessor or balance. Prioritizing decision evidence delivers both. At 48 tokens, none of the conditions delivers all required evidence in any scenario. Travel expansion excludes the unrelated hotel row and the other trip's Emma. The exact contexts, selected IDs, omissions, and pinned tokenizer hash are in `retrieval.json`.

## Next experiments

1. **Reconstruct the decision point.** The [released shopping dataset](https://huggingface.co/datasets/ZexueHe/memoryarena/blob/da1a37c8b19280e18627ca01cf368195a5e1d92e/bundled_shopping/data.jsonl) was identified. The local download failed at the sandbox proxy. The complete action–observation prefix and runnable WebShop catalog/index are also missing locally. Obtain a verified recorded prefix or generate and pin a new prefix with the released author runtime, preserving that distinction in the run label. Freeze the exact Product 4 instruction, checkpoint, environment, tool responses, model, and limits. Score the purchase outside the actor.
2. **Run the Task 24 regression.** Compare pinned author MAGE, per-concept recall, and per-concept recall with table expansion from the same available prefix. Keep the direct-table control to test whether concepts contribute beyond scoped lookup. Report predecessor and balance delivery, considered-item confusion, actual purchase, all model calls, and tool actions. The published MAGE success establishes the regression target.
3. **Measure the parts supplied by annotations.** On additional development cases, compare pre-annotated and model-written memory. Evaluate Jev on scoped identity, confirmation, and read relevance as separate decisions. Include ambiguous or unresolvable references, canceled or replaced actions, and missing evidence. Test whether concept routing locates implicit dependencies when the table and required rows must be inferred. Keep direct table lookup competitive under the same information and budgets.
4. **Run paired, held-out shopping and travel bundles.** Pin the [official MAGE source](https://github.com/Komorebi660/MAGE/tree/ace693139dc43e8bdb3076da4127483b41a5060f) and use the same model, environment, action budgets, and causal visibility. The [released shopping evaluator](https://github.com/Komorebi660/MAGE/blob/ace693139dc43e8bdb3076da4127483b41a5060f/benchmark/evaluate.py#L75) disables final judge retries for Task 24 and other listed tasks. With retries enabled on eligible bundles, the [final judge](https://github.com/Komorebi660/MAGE/blob/ace693139dc43e8bdb3076da4127483b41a5060f/benchmark/webshop/shop_mage_agent.py#L872) checks target ASINs to decide whether to attempt a revision. Run an evaluator-only-answer comparison with that trigger disabled. Track a separately labeled release-default arm if reproducing the published protocol. Predeclare the task-group split, sample size, and success criterion. Report paired wins and regressions, uncertainty on task-success differences, task progress, and total inference cost including memory operations and retries.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m experiments.concept_subspaces.run --out reports/concept_subspaces/retrieval.json
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider experiments/concept_subspaces tests/test_mage_reference.py experiments/mage_baseline/test_runtime.py -q
```
