# Limitations paragraph: delirium outcome definition and diagnosis timing

*Drafted for insertion into the Limitations section of the manuscript
(AGGP-D-26-00016). Cite/paraphrase as appropriate for house style.*

---

The delirium outcome in this study was defined as **any delirium diagnosis
code recorded during the index admission**, rather than restricting to
delirium diagnosed after a defined post-admission window. We initially
explored classifying delirium as incident (onset >48 hours after admission)
versus prevalent/present-on-admission (onset ≤48 hours, or before admission),
using the interval between each diagnosis record's timestamp
(*Diagnosis Dt Tm*) and the patient's admission datetime, in order to reduce
the risk of incorporation bias from predicting delirium that had, in effect,
already occurred. However, on inspection this timing split could not be
reliably determined from the available data. The diagnosis-type categories
that account for the large majority of delirium codes (Principal Diagnosis
and Additional Diagnosis) frequently carry a *Diagnosis Dt Tm* close to the
discharge date rather than to plausible clinical onset: 58.7% of these
records were timestamped within 24 hours of discharge, with a median gap of
only 9.7 hours before discharge, consistent with the diagnosis having been
formally coded at or near discharge-summary completion rather than
timestamped to the point of clinical recognition. Restricting the timing
analysis to diagnosis-type categories with more plausible real-time
documentation (Clinical Diagnosis, Complication) removed the great majority
of delirium cases and left an insufficient sample (68 diagnosis records) to
support a meaningful timing-based split. Because the coding practices
underlying the diagnosis timestamp field could not be disentangled from true
clinical onset with the data available, we concluded that a reliable
admission-relative timing threshold could not be established, and retained
the original, timing-agnostic outcome definition (any delirium diagnosis
during the encounter) used throughout the rest of the analysis. This is a
limitation of using retrospectively coded diagnosis timestamps to infer
clinical timing, and readers should interpret the delirium outcome as
reflecting diagnosed delirium during the admission rather than confirmed
new-onset (incident) delirium specifically. Future work with access to
nursing-documented delirium screening timestamps (e.g., CAM or 4AT
administration times) rather than coded diagnosis timestamps would be better
placed to support a reliable incident-delirium definition.
