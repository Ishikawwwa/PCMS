"""Malformed model JSON must become a logged parser failure, not an exception."""

import json

import pytest

from constraint_memory.extraction import extract


class ExtractionResponse:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, *args, **kwargs):
        return {"choices": [{"message": {"content": json.dumps(self.payload)}}]}


@pytest.mark.parametrize("payload", [
    {"unresolved": None},
    {"constraints": [{"source_refs": ["event:request"], "evidence_quote": 17}]},
    {"constraints": [{"source_refs": [{"not": "a reference"}], "evidence_quote": "Bob"}]},
])
def test_malformed_extraction_field_types_are_handled_as_parser_failures(payload):
    result = extract(ExtractionResponse(payload), {"traveler": "Bob"},
                     [{"event_id": "request", "content": "Bob requests breakfast"}])
    assert not result["parse_success"]
    assert not result["constraints"]


def test_live_malformed_scope_subject_and_target_do_not_count_as_successful_extraction():
    payload={"constraints":[{"kind":"FACTUAL","scope":"group","subject":"attribute","operator":"LTE",
        "target":{"value":3100,"unit":"USD"},"source_refs":["event:request"],"evidence_quote":"budget is $3,100"}]}
    result=extract(ExtractionResponse(payload),{"traveler":"Bob"},[{"event_id":"request","content":"The budget is $3,100."}])
    assert not result["parse_success"] and not result["constraints"]
    assert result["unresolved"][0]["reason"]=="MALFORMED_CONSTRAINT_STRUCTURE"


def test_strict_schema_is_requested_and_nullable_defaults_do_not_change_relations():
    class Client:
        def complete(self,messages,**kwargs):
            assert kwargs["response_schema"]["additionalProperties"] is False
            assert kwargs["response_schema"]["properties"]["constraints"]["items"]["properties"]["subject"]["type"]=="object"
            item={"kind":"RELATION","scope":{"group":"g","traveler":"Bob","day":null_value},
                "subject":{"traveler":"Bob","day":1,"slot":"dinner","attribute":"price"},"operator":"LTE",
                "target":{"ref":"decision:Alice:day1:dinner:price","scale":0.8,"offset":null_value,"tolerance":null_value,"unit":null_value},
                "source_refs":["event:request"],"evidence_quote":"Bob pays 20% less than Alice"}
            return {"choices":[{"message":{"content":json.dumps({"constraints":[item]})}}]}
    null_value=None
    result=extract(Client(),{"group_id":"g","traveler":"Bob"},[{"event_id":"request","content":"Bob pays 20% less than Alice"}])
    assert result["parse_success"]
    assert result["constraints"][0]["target"]=={"ref":"decision:Alice:day1:dinner:price","scale":0.8}
