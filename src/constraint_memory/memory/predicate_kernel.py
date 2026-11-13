"""Domain-neutral predicates over exact scalar/set values.

Identifiers, field names and natural-language interpretation belong to domain
adapters. This kernel has no traveler, day, slot, city, dataset or answer logic.
Unsupported types return unknown, and arithmetic uses exact decimal values.
"""
from __future__ import annotations

from decimal import Decimal,InvalidOperation,localcontext


def _equal(left,right):
    if type(left) is not type(right):
        if isinstance(left,(int,float,Decimal)) and not isinstance(left,bool) and isinstance(right,(int,float,Decimal)) and not isinstance(right,bool):
            a,b=number(left),number(right)
            return a is not None and b is not None and a==b
        return False
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(_equal(left[key],right[key]) for key in left)
    if isinstance(left,(list,tuple)):
        return len(left)==len(right) and all(_equal(a,b) for a,b in zip(left,right))
    if isinstance(left,(set,frozenset)):
        return len(left)==len(right) and all(any(_equal(a,b) for b in right) for a in left)
    return left==right


OPERATORS=frozenset({"EQUALS","NOT_EQUALS","LT","LTE","GT","GTE","IN","NOT_IN",
                     "ABS_DIFF_LTE","INTERSECTS","SUPERSET_OF","DISJOINT"})


def number(value):
    if isinstance(value,bool) or not isinstance(value,(str,int,float,Decimal)):
        return None
    try:
        result=Decimal(str(value))
        return result if result.is_finite() else None
    except (InvalidOperation,ValueError):
        return None


def transform(value,scale=1,offset=0):
    """Exact finite decimal affine transform, or unknown above a resource cap."""
    parts=[number(item) for item in (value,scale,offset)]
    if any(item is None for item in parts): return None
    precision=8+sum(len(item.as_tuple().digits)+abs(item.as_tuple().exponent) for item in parts)
    if precision>10000: return None
    with localcontext() as context:
        context.prec=max(28,precision)
        return parts[0]*parts[1]+parts[2]


def compare(value,operator,target):
    """Return True/False/None; None means the predicate is not evaluable."""
    if operator not in OPERATORS: return None
    if operator=="ABS_DIFF_LTE":
        if not isinstance(target,dict): return None
        left,right,tolerance=number(value),number(target.get("value")),number(target.get("tolerance"))
        if left is None or right is None or tolerance is None or tolerance<0: return None
        difference=transform(left,1,right.copy_negate())
        return None if difference is None else difference.copy_abs()<=tolerance
    if operator in {"LT","LTE","GT","GTE"}:
        left,right=number(value),number(target)
        if left is None or right is None: return None
        return {"LT":left<right,"LTE":left<=right,"GT":left>right,"GTE":left>=right}[operator]
    if operator in {"INTERSECTS","SUPERSET_OF","DISJOINT"}:
        if not isinstance(value,(list,tuple,set,frozenset)) or not isinstance(target,(list,tuple,set,frozenset)):
            return None
        intersects=any(_equal(a,b) for a in value for b in target)
        return {"INTERSECTS":intersects,"SUPERSET_OF":all(any(_equal(a,b) for a in value) for b in target),"DISJOINT":not intersects}[operator]
    if operator in {"IN","NOT_IN"}:
        if not isinstance(target,(list,tuple,set,frozenset)): return None
        member=any(_equal(value,item) for item in target)
        return member if operator=="IN" else not member
    # Literal equality preserves type; True is not the numeric quantity 1.
    equal=_equal(value,target)
    return equal if operator=="EQUALS" else not equal
