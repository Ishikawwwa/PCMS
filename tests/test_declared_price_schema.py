import copy

import pytest

from constraint_memory.benchmark_adapter.property_schema import annotate_price_facts,declared_price_schema


def test_unit_is_explicit_shared_schema_and_never_claims_upstream_certification():
    schema=declared_price_schema("USD")
    assert schema["currency_verified_from_upstream"] is False
    original={"decisions":[{"attribute":"price","value":37.5,"source_refs":["event:row"]},
                            {"attribute":"rating","value":4,"source_refs":["event:row"]}]}
    untouched=copy.deepcopy(original)
    result=annotate_price_facts(original,{"event_id":"scope:0","type":"property_schema","content":schema})
    price=result["decisions"][0]
    assert price["value"]==37.5 and price["unit"]=="USD"
    assert price["unit_source_kind"]=="DECLARED_PROPERTY_SCHEMA"
    assert price["source_refs"]==["event:row","event:scope:0"]
    assert price["unit_source_refs"]==["event:scope:0"]
    assert "unit" not in result["decisions"][1]
    assert original==untouched


def test_undeclared_original_facts_remain_unitless():
    original={"decisions":[{"attribute":"price","value":10}]}
    assert annotate_price_facts(original,None)==original
    assert "unit" not in annotate_price_facts(original,None)["decisions"][0]


@pytest.mark.parametrize("unit",[None,"EUR","usd",""])
def test_travel_schema_does_not_infer_currency(unit):
    with pytest.raises(ValueError): declared_price_schema(unit)


def test_schema_provenance_must_be_observed_schema_event():
    with pytest.raises(ValueError):
        annotate_price_facts({}, {"event_id":"bad","type":"model_output","content":declared_price_schema("USD")})
