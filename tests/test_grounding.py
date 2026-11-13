from constraint_memory.extraction import grounded_attributes,parse_submission


def test_attribute_grounding_uses_submitted_entity_and_city_only():
    selected=parse_submission("=== Bob's Plan ===\nDay 2:\nAccommodation: Actual Hotel, Boston")
    rows=[{"event_id":"tool1","tool_name":"AccommodationSearch","row":{"NAME":"Alternative Hotel","city":"Boston","price":20,"review rate number":5}},
          {"event_id":"tool2","tool_name":"AccommodationSearch","row":{"NAME":"Actual Hotel","city":"Boston","price":80,"review rate number":3}}]
    result=grounded_attributes(selected,rows)
    attrs={d["attribute"]:d["value"] for d in result["decisions"]}
    assert attrs=={"price":80,"rating":3}
    assert all(d["source_refs"][0]=="event:tool2" for d in result["decisions"])


def test_same_name_wrong_city_and_ambiguous_rows_are_rejected():
    selected=parse_submission("=== Bob's Plan ===\nDay 2:\nDinner: Name, Boston")
    wrong=[{"event_id":"x","tool_name":"RestaurantSearch","row":{"Name":"Name","City":"Seattle","Average Cost":10}}]
    assert not grounded_attributes(selected,wrong)["decisions"]
    rows=[{"event_id":"x","tool_name":"RestaurantSearch","row":{"Name":"Name","City":"Boston","Average Cost":10}},
          {"event_id":"y","tool_name":"RestaurantSearch","row":{"Name":"Name","City":"Boston","Average Cost":20}}]
    result=grounded_attributes(selected,rows)
    assert not result["decisions"] and result["unresolved"][0]["reason"]=="AMBIGUOUS_TOOL_ENTITY"


def test_literal_comma_in_entity_and_observed_flight_price():
    selected=parse_submission("=== A's Plan ===\nDay 1:\nAccommodation: Nice, Quiet Home, Boston\nTransportation: Flight Number: F123, from A to B, Departure Time: 09:00, Arrival Time: 10:00")
    rows=[{"event_id":"h","tool_name":"AccommodationSearch","row":{"NAME":"Nice, Quiet Home","city":"Boston","price":30}},
          {"event_id":"f","tool_name":"FlightSearch","row":{"Flight Number":"F123","Price":42}}]
    assert [d["value"] for d in grounded_attributes(selected,rows)["decisions"]]==[30,42]
