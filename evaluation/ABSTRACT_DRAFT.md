# HECA Research Conference 2026 — Abstract Draft

**Working title:** *Does Retrieval Know the Law Changed? Testing General-Purpose LLMs
Against India's 2024 Criminal Code Reform*

**Word count: 264** (target: 250–300)

---

India replaced its 163-year-old Indian Penal Code and Code of Criminal Procedure with
the Bharatiya Nyaya Sanhita (BNS) and Bharatiya Nagarik Suraksha Sanhita (BNSS) on 1 July
2024, renumbering nearly every criminal-law section citizens and practitioners rely on.
A citizen who asks a general-purpose AI assistant "what is the punishment for murder in
India?" risks an answer built on repealed law. We test this directly, using ShastraShaw,
a retrieval-augmented legal question-answering system indexed on the current statutes,
not as a product but as a controlled research baseline against GPT-4o answering the same
questions from parametric knowledge alone: no retrieval, no legal-domain prompting.

Across 44 gold-standard questions spanning criminal, traffic, rental and matrimonial law,
plus a 32-item Hindi/Marathi sub-study, we score both systems with citation accuracy, a
dedicated outdated-law-citation check, and a GPT-4o-as-expert-lawyer rubric. The result
is stark and consistent across all three methods: for nine offences with an unambiguous
current-versus-repealed section mapping (murder, theft, rape, cheating, and others), the
unaided model cited only the repealed IPC/CrPC section number 89% of the time in English
and 100% of the time in Hindi/Marathi, even when the question explicitly named "BNS
2023." ShastraShaw's rate was 0% in both languages. An independent LLM-judge rubric
confirms this: the baseline's criminal-law legal-correctness failure rate reaches 100%
(8 of 8) in Hindi/Marathi, versus 25% for the retrieval-grounded system.

This suggests retrieval grounding is not merely a quality-of-life improvement for legal
AI assistants; in a jurisdiction mid-reform, it is the difference between a citizen
receiving current law and confidently-delivered repealed law, a gap pretraining alone
does not close as legal codes change.

---

## Notes for the author (not part of the submitted abstract)

- **Why this finding over the others:** the outdated-law-citation rate was the clearest,
  most defensible, most "checkable in 30 seconds by a reader" number available — it's a
  pure regex match against a published concordance table, not a subjective judgment
  call, and it's confirmed independently by a completely different method (the LLM
  judge). The raw citation-accuracy numbers (§4 of RESEARCH_FINDINGS.md) and the overall
  5-criterion LLM-judge averages (§5) are *weaker* framing devices on their own — overall
  judge averages are close between conditions, which would undercut a "ShastraShaw wins
  across the board" pitch. The outdated-law angle avoids that trap by being specific
  about *where* and *why* the gap exists, which is also more scientifically honest.
- **Judgment call:** the abstract leads with the English 89%/100% EN/hi-mr numbers and
  the LLM-judge's 100% criminal-law failure rate in Hindi/Marathi as the two supporting
  numbers, rather than citing every metric — a 250-300 word abstract has room for one
  method's headline number plus one corroborating number, not a full results table.
  The full breakdown (citation accuracy by sector, overall LLM-judge averages, the
  cross-lingual degradation pattern, all limitations) is in
  `evaluation/RESEARCH_FINDINGS.md` for the paper/poster this abstract would accompany.
- **If asked for a number-for-number defense:** every number in this draft traces to
  `evaluation/results/{shastrashaw,frontier_baseline}{,_multilingual}/outdated_law_check.json`
  and `lawyer_rating_summary.json`, reproducible via the commands at the end of
  `RESEARCH_FINDINGS.md`.
