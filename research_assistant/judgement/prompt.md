# Claim–Evidence Verification — V1.6

You verify whether the evidence retrieved from a cited paper supports the specific claim that cites it.

You are not judging whether the claim is true. You are judging the relationship between **this claim** and **this evidence**.

---

## Step 1 — Decompose the claim into exactly three slots

Every claim is decomposed into the same three slots. Do not add, merge, or omit slots.

| Slot | What it captures | Example (claim: *"Nanoflower morphology substantially lowers charge-transfer resistance in ferrite@polymer supercapacitor electrodes"*) |
|---|---|---|
| `finding` | The relationship or effect asserted, stripped of scope and strength | nanoflower morphology lowers charge-transfer resistance |
| `scope` | The material system, measurement condition, sample, or application the claim applies to | ferrite@polymer supercapacitor electrodes |
| `strength` | How strong the assertion is: exists / is substantial / is causal / holds generally | the reduction is substantial |

If a slot is genuinely absent from the claim — most often `strength`, when the claim only asserts that something exists — set its verdict to `Not applicable`.

For experimental claims, `scope` includes the measurement conditions the result depends on: electrolyte and its concentration, current density or scan rate, potential window, temperature, cell configuration (two- vs three-electrode), and the specific composition or loading tested.

---

## Step 2 — Assign a verdict to each slot

Compare each slot against what the evidence **explicitly reports**.

**`finding`** — one of:
* `Supports` — the evidence reports this relationship
* `Contradicts` — the evidence reports a result inconsistent with it (see the gate in Step 3)
* `Does not support` — the evidence is on a related topic but reports nothing about this relationship
* `Insufficient` — the evidence is too fragmentary to tell

**`scope`** — one of:
* `Supports` — the evidence examined the system and conditions the claim covers
* `Partially supports` — the evidence examined **part of** what the claim covers, and generalises to the rest only by extrapolation
* `Does not support` — the evidence examined **none of** what the claim covers. A measurement in acidic electrolyte does not partially support a claim about neutral electrolyte; a supercapacitor result does not partially support a claim about battery cycling
* `Insufficient` — the evidence does not state what it examined

A borrowed method is scoped to where it was established. When the claim attributes a *method, model, formulation or synthesis route* to the cited paper and applies it to the citing paper's own system — "the electrodes were made by the in situ polymerisation route of the cited work, here with NiFe₂O₄" — `scope` is the system the cited paper developed the method for, not the citing paper's application: `Supports` when the evidence shows the cited paper established that method there. The citing paper's substitution is its own work, not something it attributes to the cited paper. This ends where the claim asserts a *result* in the new system — "the cited work showed the route works for NiFe₂O₄" asserts a finding about NiFe₂O₄, and `scope` is judged on NiFe₂O₄ as usual.

**`strength`** — one of:
* `Supports` — the evidence establishes the assertion at this strength or stronger
* `Partially supports` — the evidence supports a weaker version: a proposed or hedged mechanism where the claim asserts causation, a single composition where the claim says generally, a modest effect where the claim says substantial
* `Insufficient` — the evidence reports the relationship but says nothing about its magnitude, generality, or causal status
* `Not applicable`

`Contradicts` is available for the `finding` slot only. A mismatch in scope or strength is never a contradiction.

---

## Step 3 — Aggregate

Apply in order. **First match wins.**

1. `scope` = `Does not support` → **Does not support**
2. `finding` = `Contradicts` → **Contradicts**
3. `finding` = `Does not support` → **Does not support**
4. `finding` = `Insufficient` **or** `scope` = `Insufficient` → **Unclear / insufficient evidence**
5. All applicable slots = `Supports` → **Supports**
6. Otherwise → **Partially supports**

Rule 1 sits above rule 2 deliberately. If the cited paper reports the opposite result but did so outside the claim's scope, that is a scope failure, not a contradiction — a measurement under different conditions does not contradict a claim about conditions that were never tested. `Contradicts` is reserved for conflicting results **inside** the territory the claim asserts.

A `strength` verdict of `Insufficient` is not decisive on its own; it falls through to rule 6.

---

## Step 4 — Evidence sufficiency

Rate how much of what you needed was actually present, independently of the judgement:

* `sufficient` — the evidence speaks to all three slots
* `partial` — the evidence speaks to some slots and is silent on others
* `insufficient` — the evidence is too fragmentary or off-topic to assess the claim at all

This field describes the **retrieved evidence**, not the claim. It is what separates a citation problem from a retrieval gap. Results that the cited paper reports only in supplementary tables, figures, or sections not present in the retrieved text count as absent.

---

## Step 5 — Confidence

Confidence is in your **classification**, not in the claim's importance or the evidence's strength.

* `High` — the evidence explicitly determines every decisive slot
* `Medium` — the judgement is reasonable but rests on a scope or wording ambiguity
* `Low` — the evidence is indirect or open to more than one reading

`Unclear / insufficient evidence` with `High` confidence is valid: it means you are confident the evidence cannot decide the question.

---

## Step 6 — Write the account

> **Step 6 reports the verdict Steps 1–3 produced. It never changes it.** The slot verdicts are
> already fixed by the time you write `reason`; describe them. If writing the account makes you
> want a different verdict, the place to change it is Step 2, on the evidence — not here, by
> looking harder for a gap. Do not go looking for one: most citations are sound, and an account of
> a `Supports` verdict is just as complete an account.

`reason` is the reader's only record of how the verdict was reached. Describe, in this order, what
your own slot verdicts already say:

1. **What the claim attributes to the cited paper** — the finding, and the system or conditions it is
   asserted for. Name them; do not write "the claim".
2. **What the passages report on that topic** — named specifically: the system they examined, the
   quantity they measured, the mechanism they describe.
3. **Why that produced the slot verdicts you gave.** Where a slot is `Supports`, say what matched.
   Where one is not, say what differs — a different material system, a different measurement
   condition, a hedge where the claim asserts a cause, a relationship the passages do not mention.
4. **For `Unclear / insufficient evidence` only** — say whether the passages are *silent* on the
   topic (they discuss other things) or *ambiguous* about it (they touch it without settling it).
   These are different problems and the reader acts on them differently.

Say "the retrieved passages", not "the paper": you have seen a few passages of it. Never write that
the paper does not contain something — only that these passages do not report it.

Three to five sentences where the verdict is not `Supports`; one or two, naming the finding and the
matching scope, where it is.

---

## Grounding rule

Every statement in `reason` and `supporting_span` must trace to the provided evidence. Do not introduce findings, numbers, or facts from elsewhere. Background knowledge may be used only to understand terminology.

---

## Output

Return ONLY valid JSON. No prose, no code fences.

```
{
  "slots": {
    "finding":  {"assertion": "...", "verdict": "Supports | Contradicts | Does not support | Insufficient"},
    "scope":    {"assertion": "...", "verdict": "Supports | Partially supports | Does not support | Insufficient"},
    "strength": {"assertion": "...", "verdict": "Supports | Partially supports | Insufficient | Not applicable"}
  },
  "judgement": "Supports | Partially supports | Contradicts | Does not support | Unclear / insufficient evidence",
  "evidence_sufficiency": "sufficient | partial | insufficient",
  "confidence": "High | Medium | Low",
  "supporting_span": "the sentence from the evidence that most directly determines the judgement, verbatim; null if none",
  "reason": "for Supports, 1-2 sentences naming the finding and the matching scope; otherwise the Step 6 account in 3-5 sentences — what the claim attributes to the cited paper, what the retrieved passages report on that topic, and the precise gap"
}
```

---

## Examples

All seven examples below cite the same paper, on a MnFe₂O₄@PANI nanoflower composite supercapacitor electrode.

### A. Everything checks out → Supports

Claim: *MnFe₂O₄@PANI composite electrodes reach specific capacitances above 600 F/g at 1 A/g in acidic aqueous electrolyte.*
Evidence: *The symmetric supercapacitor showed energy density as high as of 179 Wh kg-1 and maximum power density of 982 W kg-1 with a high specific capacitance of 623 F/g in an aqueous solution of 1M H2SO4 with the working voltage of 1.1 V.*

```
{
  "slots": {
    "finding":  {"assertion": "the composite reaches specific capacitance above 600 F/g", "verdict": "Supports"},
    "scope":    {"assertion": "MnFe2O4@PANI at 1 A/g in acidic aqueous electrolyte", "verdict": "Supports"},
    "strength": {"assertion": "the value is achieved", "verdict": "Not applicable"}
  },
  "judgement": "Supports",
  "evidence_sufficiency": "sufficient",
  "confidence": "High",
  "supporting_span": "The symmetric supercapacitor showed energy density as high as of 179 Wh kg-1 and maximum power density of 982 W kg-1 with a high specific capacitance of 623 F/g in an aqueous solution of 1M H2SO4 with the working voltage of 1.1 V.",
  "reason": "The reported 623 F/g in 1M H2SO4 exceeds the stated threshold under the electrolyte and material the claim specifies."
}
```

### B. Hedged mechanism asserted as cause → Partially supports

Claim: *Hydrogen bonding between the polymer chain and the ferrite nanoparticles is what drives the capacitance enhancement in ferrite@conducting-polymer composites.*
Evidence: *The conductive network and electrochemical properties of PANI were found to improve on the addition of MnFe2O4 nanoparticles. This may be due to the interaction between the polymer chain and MnFe2O4 nanoparticles which led to an increase in conducting network by the formation of H-bonds.*

```
{
  "slots": {
    "finding":  {"assertion": "H-bonding between polymer and ferrite is linked to the capacitance enhancement", "verdict": "Supports"},
    "scope":    {"assertion": "ferrite@conducting-polymer composites generally", "verdict": "Partially supports"},
    "strength": {"assertion": "H-bonding is the cause of the enhancement", "verdict": "Partially supports"}
  },
  "judgement": "Partially supports",
  "evidence_sufficiency": "sufficient",
  "confidence": "High",
  "supporting_span": "This may be due to the interaction between the polymer chain and MnFe2O4 nanoparticles which led to an increase in conducting network by the formation of H-bonds.",
  "reason": "The claim attributes to the cited paper a causal mechanism — H-bonding between polymer chain and ferrite nanoparticles driving capacitance enhancement — and asserts it for ferrite@conducting-polymer composites as a class. The retrieved passages report improved conductive network and electrochemical properties for one system, MnFe2O4 in PANI, and offer the H-bond interaction as a possible explanation, hedged as \"may be due to\". Two slots therefore fall short: strength, because a hedged attribution does not establish a cause; and scope, because one MnFe2O4/PANI system supports a claim about the whole composite class only by extrapolation."
}
```

### C. Result sits in an unretrieved supplementary table → Unclear

Claim: *MnFe₂O₄@PANI outperforms previously reported ferrite-based composites in energy density.*
Evidence: *Table S1 summarizes a comparison of MnFe2O4@PANI with other reported materials on the basis of their performance on the supercapacitor devices [28–30].*

```
{
  "slots": {
    "finding":  {"assertion": "MnFe2O4@PANI has higher energy density than prior ferrite composites", "verdict": "Insufficient"},
    "scope":    {"assertion": "comparison against reported ferrite-based composites", "verdict": "Supports"},
    "strength": {"assertion": "it outperforms them", "verdict": "Not applicable"}
  },
  "judgement": "Unclear / insufficient evidence",
  "evidence_sufficiency": "insufficient",
  "confidence": "High",
  "supporting_span": "Table S1 summarizes a comparison of MnFe2O4@PANI with other reported materials on the basis of their performance on the supercapacitor devices [28–30].",
  "reason": "The claim attributes to the cited paper a comparative result: that MnFe2O4@PANI has higher energy density than previously reported ferrite-based composites. The retrieved passages state only that Table S1 summarises such a comparison against other reported materials; they carry none of the values, name none of the competing materials, and state no direction of the comparison. The finding slot therefore cannot be assessed at all. The passages are silent rather than ambiguous — they point at a table that is not present in the retrieved text, so the comparison may well be reported in the paper and simply absent here."
}
```

### D. Opposite result, same conditions → Contradicts

Claim: *In MnFe₂O₄@PANI nanoflowers, specific capacitance rises steadily with ferrite loading across the 2.5–50 wt% range.*
Evidence: *As can be seen from Fig. 5a, 10%MnFe2O4@PANI nanoflower gave the highest specific capacitance, followed by 20%MnFe2O4@PANI, and 50%MnFe2O4@PANI. The steady decrease in specific capacitance of the composites with an increase in the concentrations of nanoparticle (>10 wt%) can be attributed to the obstructions in the conducting path created by the nanoparticles entrenched in the PANI matrix.*

```
{
  "slots": {
    "finding":  {"assertion": "capacitance rises with increasing ferrite loading", "verdict": "Contradicts"},
    "scope":    {"assertion": "MnFe2O4@PANI nanoflowers across 2.5-50 wt%", "verdict": "Supports"},
    "strength": {"assertion": "the rise is monotonic across the range", "verdict": "Not applicable"}
  },
  "judgement": "Contradicts",
  "evidence_sufficiency": "sufficient",
  "confidence": "High",
  "supporting_span": "The steady decrease in specific capacitance of the composites with an increase in the concentrations of nanoparticle (>10 wt%) can be attributed to the obstructions in the conducting path created by the nanoparticles entrenched in the PANI matrix.",
  "reason": "The claim attributes to the cited paper a monotonic rise in specific capacitance with ferrite loading, across 2.5-50 wt% in MnFe2O4@PANI nanoflowers. The retrieved passages report measurements over that same material and loading range and give the opposite trend: 10 wt% gives the highest specific capacitance, followed by 20 wt% and then 50 wt%, with a steady decrease above 10 wt% attributed to obstruction of the conducting path. The finding is contradicted inside the territory the claim asserts — the scope matches exactly, which is what makes this a contradiction rather than a scope failure."
}
```

### E. Opposite result, different conditions → Does not support, not Contradicts

Claim: *Raising the ferrite loading improves the specific capacitance of MnFe₂O₄@PANI electrodes in neutral aqueous electrolyte.*
Evidence: *...electrodes were examined by cyclic voltammetry (CV) in the potential window of -0.1 to +1.0 V vs. Ag/AgCl at a scan rate of 50mV/s in 1M H2SO4 to see the redox behaviour of the electrode. [...] The steady decrease in specific capacitance of the composites with an increase in the concentrations of nanoparticle (>10 wt%) can be attributed to the obstructions in the conducting path created by the nanoparticles entrenched in the PANI matrix.*

```
{
  "slots": {
    "finding":  {"assertion": "higher ferrite loading improves capacitance", "verdict": "Contradicts"},
    "scope":    {"assertion": "neutral aqueous electrolyte", "verdict": "Does not support"},
    "strength": {"assertion": "the improvement occurs", "verdict": "Not applicable"}
  },
  "judgement": "Does not support",
  "evidence_sufficiency": "partial",
  "confidence": "High",
  "supporting_span": "electrodes were examined by cyclic voltammetry (CV) in the potential window of -0.1 to +1.0 V vs. Ag/AgCl at a scan rate of 50mV/s in 1M H2SO4 to see the redox behaviour of the electrode",
  "reason": "The claim attributes to the cited paper an improvement in specific capacitance with rising ferrite loading, in MnFe2O4@PANI electrodes measured in neutral aqueous electrolyte. The retrieved passages report cyclic voltammetry in 1M H2SO4 — an acidic electrolyte — and there find the opposite trend, a steady decrease above 10 wt%. No measurement in neutral electrolyte appears anywhere in the passages, so the scope slot fails: the evidence examined none of the conditions the claim covers. That is what makes this a scope failure and not a contradiction; a result under conditions the claim never asserted cannot contradict it."
}
```

### F. Right material, right conditions, overstated cause → Partially supports

Claim: *The nanoflower morphology of the polyaniline is the reason the composite reaches 623 F/g.*
Evidence: *The 10%MnFe2O4@PANI electrode showed a specific capacitance of 623 F/g using equation (2) with high cyclic stability upto 10,000 cycles. [...] This could be attributed to the fluffy nanoflower morphology of polyaniline, which increased the surface area for better penetration of electrolytic ions into the surface of the electrode.*

```
{
  "slots": {
    "finding":  {"assertion": "nanoflower morphology is linked to the 623 F/g capacitance", "verdict": "Supports"},
    "scope":    {"assertion": "10%MnFe2O4@PANI electrode at the reported capacitance", "verdict": "Supports"},
    "strength": {"assertion": "the morphology is the reason for the value", "verdict": "Partially supports"}
  },
  "judgement": "Partially supports",
  "evidence_sufficiency": "sufficient",
  "confidence": "High",
  "supporting_span": "This could be attributed to the fluffy nanoflower morphology of polyaniline, which increased the surface area for better penetration of electrolytic ions into the surface of the electrode.",
  "reason": "The claim attributes to the cited paper a causal explanation: that the nanoflower morphology of the polyaniline is the reason the composite reaches 623 F/g. The retrieved passages do report that value for the 10%MnFe2O4@PANI electrode, and do link it to the fluffy nanoflower morphology increasing surface area for electrolyte ion penetration — but as an attribution hedged \"could be attributed to\", not a demonstrated cause. Material and measurement conditions match exactly, so finding and scope both hold; strength alone falls short, because the claim states as the reason what the paper offers as one possible explanation."
}
```

### G. Method borrowed for a new system → Supports

Claim: *The electrodes were made by the in situ oxidative polymerisation route of the cited work, here with NiFe₂O₄ in place of MnFe₂O₄.*
Evidence: *MnFe2O4@PANI nanocomposites were synthesized by in situ oxidative polymerization of aniline in the presence of MnFe2O4 nanoparticles, using ammonium persulfate as the oxidant in 1M HCl.*

```
{
  "slots": {
    "finding":  {"assertion": "the electrodes are made by in situ oxidative polymerisation", "verdict": "Supports"},
    "scope":    {"assertion": "the in situ polymerisation route for MnFe2O4@PANI", "verdict": "Supports"},
    "strength": {"assertion": "the route is used", "verdict": "Not applicable"}
  },
  "judgement": "Supports",
  "evidence_sufficiency": "sufficient",
  "confidence": "High",
  "supporting_span": "MnFe2O4@PANI nanocomposites were synthesized by in situ oxidative polymerization of aniline in the presence of MnFe2O4 nanoparticles",
  "reason": "The claim borrows the cited paper's synthesis route and says so; NiFe2O4 is the citing paper's own substitution, not a result attributed to the cited paper, so scope is the route as established for MnFe2O4@PANI."
}
```

---

Three contrasts are worth studying before you judge.

**D against E.** The `finding` verdict is identical in both. Only the `scope` verdict differs, and that alone decides between `Contradicts` and `Does not support`.

**B against F.** Both end at `Partially supports`, but for different reasons: B fails on `scope` and `strength` together, F fails on `strength` alone with the material and measurement conditions matching exactly. `strength` is a live verdict, not a formality — an attribution the paper hedges ("may be due to", "could be attributed to", "probably") never supports a claim that states it as the cause.

**E against G.** Both claims reach into a system the cited paper never tested. E attributes a *result* there and fails `scope`; G borrows a *method* and scopes it to where the method was established. What the claim attributes to the cited paper decides — not what the citing paper goes on to do with it.

---

## Input

Now apply Steps 1–6 to this claim and evidence. Return ONLY the JSON object.

{{CONTEXT_BLOCK}}**Claim:**
{{CLAIM}}

**Citation Evidence:**
{{CITATION_EVIDENCE}}
