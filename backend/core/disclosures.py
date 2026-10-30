"""
Disclosures the product owes the user, kept apart from prose an LLM writes.

Three facts about a hedge candidate are statements the product makes, not
commentary on it: that its premium is a model estimate rather than a price
anyone will trade at, that its market is too thin to rely on, and that its
expiry crosses an earnings announcement. Each one changes what the numbers
beside it mean, so each has to reach the user.

They were being carried in `candidate.cons` — which is also where the LLM
explainer writes. Both explainer layers merge a response by *replacing* the
list outright:

    if explanation.cons:
        candidate.cons = explanation.cons

and L11 runs by default (`run_l11_llm = True`). So a model that returned its
own three bullets deleted the executability disclosure on the way past, and
the response went out showing a theoretical premium with nothing on screen
saying it was theoretical. The disclosure was not weakened or argued with —
it was gone, which is worse, because its absence reads as confirmation that
the premium is a real one.

The fix leaves `cons` as the single thing the UI renders, so nothing about
the display changes, and adds a second record of what the product itself
said that the LLM has no path to. `disclose()` writes both; `restore()`,
called after every merge, puts back whatever the merge dropped.

Scope is deliberately the three hard disclosures and not every sentence the
product generates. Rewriting the basis-R² prose in `rationale` into better
English is the explainer's job and loses nothing material; dropping "this
price is not real" is a different kind of event.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # pragma: no cover
    from backend.models.hedge_models import InstrumentCandidate


def disclose(candidate: "InstrumentCandidate", text: str) -> None:
    """
    State `text` as a product disclosure on `candidate`.

    Appends to `cons` exactly as the direct assignment it replaces did, and
    records the same text where a later overwrite of `cons` cannot reach it.
    Idempotent: enriching a candidate twice does not say the same thing twice.
    """
    if text not in candidate.mandatory_disclosures:
        candidate.mandatory_disclosures = list(candidate.mandatory_disclosures) + [text]
    if text not in candidate.cons:
        candidate.cons = list(candidate.cons) + [text]


def restore(candidate: "InstrumentCandidate") -> None:
    """
    Put back any disclosure a write to `cons` dropped.

    Restored disclosures go in front of the surviving cons rather than on the
    end. A caveat the user has to scroll past four bullets of LLM prose to
    find is a caveat they will not read, and these are the ones that decide
    whether the premium above them is a number at all.

    Called after every merge rather than only when one looks necessary: the
    check is a list comparison, and a merge path that forgets to ask is
    exactly how this defect arrived.
    """
    missing = [
        d for d in candidate.mandatory_disclosures
        if d not in candidate.cons
    ]
    if missing:
        candidate.cons = missing + list(candidate.cons)
