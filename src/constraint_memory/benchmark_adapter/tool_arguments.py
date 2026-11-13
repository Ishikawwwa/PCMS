"""Declared shared argument adapter for public CSV city-search tools."""
import re

from constraint_memory.memory.grounding import US_STATE_ANNOTATIONS


CITY_TOOLS=frozenset({"RestaurantSearch","AccommodationSearch","AttractionSearch"})
TOOL_ARGUMENT_CONTRACT_TEXT=(
    "PUBLIC CSV TOOL CONTRACT: search tools use plain city names. The shared adapter "
    "removes a recognized trailing US state annotation from RestaurantSearch, "
    "AccommodationSearch and AttractionSearch city arguments, for example "
    "San Antonio(Texas) becomes San Antonio. Other arguments and tools remain literal. "
    "For a newly chosen restaurant or accommodation, copy the full Name from the returned "
    "row, including colon suffixes; an abbreviated property title cannot identify the exact row."
)


def normalize_city_arguments(tool_name,arguments,enabled=False):
    if not isinstance(arguments,dict):
        raise ValueError("Official tool arguments must be an object")
    result=dict(arguments)
    changes={}
    city=result.get("city")
    if enabled and tool_name in CITY_TOOLS and isinstance(city,str):
        match=re.fullmatch(r"([^()]+?)\s*\(([^()]+)\)",city)
        if match and match[2].strip().casefold() in US_STATE_ANNOTATIONS and match[1].rstrip():
            effective=match[1].rstrip()
            result["city"]=effective
            changes["city"]={"original":city,"effective":effective,"rule":"recognized_us_state_annotation"}
    return result,changes
