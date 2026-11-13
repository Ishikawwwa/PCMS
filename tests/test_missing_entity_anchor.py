from constraint_memory.memory import create_backend


def test_missing_prior_price_still_returns_exact_entity_to_query():
    memory=create_backend("E3",{"token_budget":10000}); memory.reset("g")
    memory.begin_subtask({"group_id":"g","traveler":"Bob","step":0,"query":"dinner"})
    memory.commit_subtask({"group_id":"g","traveler":"Bob","step":0,"action":"Exact Cafe, Boston",
        "decisions":[{"traveler":"Bob","day":1,"slot":"dinner","value":"Exact Cafe, Boston","source_refs":["event:action"]}]})
    memory.observe({"group_id":"g","event_id":"price-request","step":1,"type":"request",
                    "content":"Carol's dinner costs at most 80% of Bob's dinner."})
    context=memory.begin_subtask({"group_id":"g","traveler":"Carol","step":1,"query":"Carol's dinner costs at most 80% of Bob's dinner.",
        "extraction":{"constraints":[{"constraint_id":"discount","kind":"RELATION","subject":{"traveler":"Carol","day":1,"slot":"dinner","attribute":"price"},
            "operator":"LTE","target":{"ref":"decision:Bob:day1:dinner:price","scale":0.8},"source_refs":["event:price-request"]}]}})
    assert context["status"]=="UNRESOLVED"
    assert "decision:Bob:day1:dinner:price" in context["missing_refs"]
    assert any(r.get("decision_id")=="decision:Bob:day1:dinner" and r["value"]=="Exact Cafe, Boston" for r in context["records"])
    assert not any(r.get("attribute")=="price" and r.get("traveler")=="Bob" for r in context["records"])


def test_missing_property_never_resurrects_withdrawn_entity():
    memory=create_backend("E3",{"token_budget":10000}); memory.reset("g")
    request={"group_id":"g","traveler":"Bob","step":0,"query":"dinner"}
    memory.begin_subtask(request)
    memory.commit_subtask({**request,"action":"Old Cafe, Boston","decisions":[{"traveler":"Bob","day":1,"slot":"dinner","value":"Old Cafe, Boston"}]})
    memory.commit_subtask({**request,"step":1,"action":"","replace_travelers":["Bob"],"decisions":[]})
    context=memory.begin_subtask({"group_id":"g","traveler":"Carol","step":2,"query":"price",
                                  "dependency_refs":["decision:Bob:day1:dinner:price"]})
    assert not any(r.get("value")=="Old Cafe, Boston" for r in context["records"])


def test_public_base_is_not_duplicated_by_a_casual_name_mention_but_typed_edges_keep_anchors():
    memory=create_backend("E3",{"token_budget":10000}); memory.reset("g")
    memory.begin_subtask({"group_id":"g","traveler":"Base","step":0})
    memory.commit_subtask({"group_id":"g","traveler":"Base","step":0,"action":"Exact Cafe, Boston",
        "decisions":[{"traveler":"Base","day":1,"slot":"dinner","value":"Exact Cafe, Boston"}]})
    casual=memory.begin_subtask({"group_id":"g","traveler":"Bob","step":1,"query":"Going with Base",
                                "public_anchor_travelers":["Base"]})
    assert not any(r.get("traveler")=="Base" for r in casual["records"])
    assert "Exact Cafe, Boston" not in casual["context"]
    explicit=memory.begin_subtask({"group_id":"g","traveler":"Bob","step":2,"query":"Going with Base",
        "public_anchor_travelers":["Base"],"dependency_refs":["decision:Base:day1:dinner:price"]})
    assert any(r.get("value")=="Exact Cafe, Boston" for r in explicit["records"])
    assert "decision:Base:day1:dinner:price" in explicit["missing_refs"]
