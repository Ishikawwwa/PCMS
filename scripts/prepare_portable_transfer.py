"""Create disclosed synthetic development inputs; no expected answers are stored."""
from __future__ import annotations
import copy
import json
import hashlib
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def cases():
    specifications=[
        ("scheduling", {"prior.finish":50,"deadline":170}, {"prior.finish":100},
         [{"subject_ref":"chosen.start","operator":"GTE","target":{"ref":"prior.finish","offset":10},"unit":"minute"},
          {"subject_ref":"chosen.end","operator":"LTE","target":{"ref":"deadline"},"unit":"minute"}],
         [{"start":95,"end":130},{"start":115,"end":160},{"start":120,"end":180}],
         {"start":"chosen.start","end":"chosen.end"},"minute","Book the follow-up using the current finish and deadline requirements."),
        ("procurement", {"budget":120,"lead.limit":4}, {"budget":90},
         [{"subject_ref":"chosen.cost","operator":"LTE","target":{"ref":"budget"},"unit":"EUR"},
          {"subject_ref":"chosen.lead","operator":"LTE","target":{"ref":"lead.limit"},"unit":"day"}],
         [{"cost":95,"lead":3},{"cost":85,"lead":3},{"cost":80,"lead":6}],
         {"cost":"chosen.cost","lead":"chosen.lead"},{"cost":"EUR","lead":"day"},"Purchase under the revised authorization and delivery requirement."),
        ("deployment", {"required.flags":["audit","tls"],"allowed.regions":["eu","us"]}, {"required.flags":["audit","tls","trace"]},
         [{"subject_ref":"chosen.flags","operator":"SUPERSET_OF","target":{"ref":"required.flags"}},
          {"subject_ref":"chosen.region","operator":"IN","target":{"ref":"allowed.regions"}}],
         [{"flags":["audit","tls"],"region":"eu"},{"flags":["audit","tls","trace"],"region":"eu"},
          {"flags":["audit","tls","trace"],"region":"apac"}],
         {"flags":"chosen.flags","region":"chosen.region"},None,"Select a deployment satisfying the current release policy."),
        ("inventory", {"reserved":3,"required.zone":"cool"}, {"reserved":7},
         [{"subject_ref":"chosen.available","operator":"GTE","target":{"ref":"reserved"},"unit":"item"},
          {"subject_ref":"chosen.zone","operator":"EQUALS","target":{"ref":"required.zone"}}],
         [{"available":6,"zone":"cool"},{"available":9,"zone":"cool"},{"available":12,"zone":"dry"}],
         {"available":"chosen.available","zone":"chosen.zone"},{"available":"item"},"Allocate an offered bin while preserving the updated reservation and zone rule.")]
    result=[]
    for domain,initial,revision,predicates,rows,bindings,units,query in specifications:
        for load in ("short","pressure","infeasible"):
            for style in ("tables","json","messages"):
                def unit(field):
                    return units.get(field) if isinstance(units,dict) else units
                def facts(values):
                    return [{"ref":ref,"pointer":"/"+ref,"unit":
                        ("EUR" if ref=="budget" else "day" if ref=="lead.limit" else
                         "item" if ref=="reserved" else "minute" if domain=="scheduling" else None)} for ref in values]
                current_revision=revision
                if load=="infeasible":
                    current_revision={"prior.finish":200} if domain=="scheduling" else {"budget":50} if domain=="procurement" else {"required.flags":["non-offered-critical"]} if domain=="deployment" else {"reserved":99}
                events=[{"event_id":"initial","step":0,"kind":"observation","content":initial,"facts":facts(initial)},
                        {"event_id":"policy","step":1,"kind":"task_contract","content":predicates,
                         "constraints":[{"ref":"policy-"+str(i),"pointer":"/"+str(i)} for i in range(len(predicates))]},
                        {"event_id":"revision","step":2,"kind":"observation","content":current_revision,"facts":facts(current_revision)}]
                if load=="pressure":
                    for index in range(60):
                        events.append({"event_id":"unrelated-"+str(index),"step":index+3,"kind":"observation",
                            "content":{"unrelated_activity":"Quarterly notes for another project. "*12,"record":index}})
                step=events[-1]["step"]+1
                observed=[{**row,"option_id":hashlib.sha256((domain+load+str(index)).encode()).hexdigest()[:20]} for index,row in enumerate(rows)]
                # Vary offered order independently of format, while preserving
                # exact content. No canonical correct option is put in inputs.
                random.Random(domain+load+style).shuffle(observed)
                events.append({"event_id":"offers","step":step,"kind":"observation","content":observed})
                options=[{"option_id":row["option_id"],"event_id":"offers","pointer":"/"+str(index),
                    "properties":[{"ref":ref,"pointer":f"/{index}/{field}","unit":unit(field)} for field,ref in bindings.items()]}
                    for index,row in enumerate(observed)]
                result.append(copy.deepcopy({"case_id":domain+"-"+load+"-"+style,"domain":domain,"events":events,
                    "active_constraint_refs":["policy-"+str(i) for i in range(len(predicates))],"options":options,
                    "query":query,"format":style}))
    return result


if __name__=="__main__":
    destination=ROOT/"data/transfer/development.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(cases(),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Prepared {len(cases())} synthetic development cases: {destination}")
