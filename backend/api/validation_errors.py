"""
Human-readable 422 bodies for request validation failures.

FastAPI's default handler returns pydantic's own error objects verbatim. For a
holding submitted without `position_type` that is:

    {"type": "union_tag_not_found",
     "loc": ["body", "holdings", 0],
     "msg": "Unable to extract tag using discriminator 'position_type'",
     "input": {"ticker": "AAPL", "quantity": 100},
     "ctx": {"discriminator": "'position_type'"}}

Three problems, all in scope for this repair. `msg` is library phrasing about
pydantic's internals, and the frontend copies it straight into the field error
it renders, so a user submitting a portfolio is shown the word "discriminator".
`type` and `ctx` expose which validation library the service is built on and
how its models are structured. And `input` echoes the submitted value back in
the response body, which for this product is position-level portfolio data
that need not travel back out to be told it was malformed.

This handler keeps `loc` exactly as pydantic produced it, because the client
uses it to highlight the offending field, and replaces everything else with a
sentence addressed to the person who filled in the form. The untranslated
original is logged server-side, so nothing is lost for debugging.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# Discriminated-union variant tags. Pydantic puts the matched variant into
# `loc` ("holdings", 0, "stock", "shares"), which is correct addressing but
# reads as a field the user never typed, so the prose skips it. `loc` itself
# still carries it, because that is what the client matches fields on.
_VARIANT_TAGS = {"stock", "option"}


def _field_name(loc: List[Any]) -> str:
    """
    "holdings[0].shares" from ["body", "holdings", 0, "stock", "shares"].

    Only used inside the message prose; `loc` is passed through untouched so
    the client's own field-matching keeps working.
    """
    parts = [p for p in loc if p != "body"]
    # Drop a variant tag only where it sits between an index and a real field,
    # so a user-facing field that happens to be called "option" survives.
    pruned: List[Any] = []
    for i, part in enumerate(parts):
        if (
            part in _VARIANT_TAGS
            and i > 0
            and isinstance(parts[i - 1], int)
            and i + 1 < len(parts)
        ):
            continue
        pruned.append(part)
    if not pruned:
        return "the request"
    out = str(pruned[0])
    for part in pruned[1:]:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


def _humanize(err: Dict[str, Any]) -> str:
    """One sentence a user can act on, for one pydantic error."""
    etype = err.get("type", "")
    ctx   = err.get("ctx") or {}
    field = _field_name(err.get("loc") or [])

    if etype == "missing":
        return f"{field} is required."

    if etype == "union_tag_not_found":
        # The tag name is the one piece of ctx worth keeping: it names a field
        # the caller actually has to supply.
        tag = str(ctx.get("discriminator", "type")).strip("'\"")
        return (
            f"{field} must include \"{tag}\" so we know which kind of entry it is."
        )

    if etype == "union_tag_invalid":
        tag      = str(ctx.get("discriminator", "type")).strip("'\"")
        expected = str(ctx.get("expected_tags", "")).strip()
        allowed  = f" Allowed values: {expected}." if expected else ""
        return f"{field} has an unrecognised \"{tag}\".{allowed}"

    if etype in ("int_parsing", "int_type"):
        return f"{field} must be a whole number."
    if etype in ("float_parsing", "float_type", "decimal_parsing"):
        return f"{field} must be a number."
    if etype in ("bool_parsing", "bool_type"):
        return f"{field} must be true or false."
    if etype in ("string_type", "str_type"):
        return f"{field} must be text."
    if etype in ("list_type", "tuple_type"):
        return f"{field} must be a list."
    if etype in ("model_type", "dict_type", "model_attributes_type"):
        return f"{field} must be an object."
    if etype == "date_parsing" or etype == "date_from_datetime_parsing":
        return f"{field} must be a date in YYYY-MM-DD form."

    if etype == "greater_than":
        return f"{field} must be greater than {ctx.get('gt')}."
    if etype == "greater_than_equal":
        return f"{field} must be at least {ctx.get('ge')}."
    if etype == "less_than":
        return f"{field} must be less than {ctx.get('lt')}."
    if etype == "less_than_equal":
        return f"{field} must be at most {ctx.get('le')}."

    if etype == "too_short":
        return f"{field} needs at least {ctx.get('min_length', 1)} entr" + (
            "y." if ctx.get("min_length", 1) == 1 else "ies."
        )
    if etype == "too_long":
        return f"{field} may have at most {ctx.get('max_length')} entries."
    if etype == "string_too_short":
        return f"{field} must be at least {ctx.get('min_length')} characters."
    if etype == "string_too_long":
        return f"{field} must be at most {ctx.get('max_length')} characters."

    if etype == "enum":
        expected = ctx.get("expected")
        return f"{field} must be one of: {expected}." if expected else f"{field} is not a permitted value."

    if etype == "extra_forbidden":
        return f"{field} is not a field this endpoint accepts."

    if etype == "value_error":
        # Raised by this project's own validators, so the text is already
        # written for a user — pydantic only prefixes it.
        msg = str(err.get("msg", "")).removeprefix("Value error, ").strip()
        return msg or f"{field} is not valid."

    if etype == "json_invalid":
        return "The request body is not valid JSON."

    # Anything unmapped: say where without quoting the library.
    return f"{field} is not valid."


def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    raw = exc.errors()

    # The untranslated errors stay in the log; `input` is the submitted value,
    # so only its shape is recorded rather than its contents.
    logger.info(
        "422 on %s %s: %s",
        request.method,
        request.url.path,
        [{"type": e.get("type"), "loc": e.get("loc")} for e in raw],
    )

    detail = [{"loc": list(e.get("loc") or []), "msg": _humanize(e)} for e in raw]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": detail},
    )
