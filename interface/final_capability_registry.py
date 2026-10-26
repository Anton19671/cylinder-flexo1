"""Elastic coupled solver capability API (no legacy material routes)."""
from dataclasses import dataclass
SUPPORTED="SUPPORTED"
LIMITED="LIMITED"
UNAVAILABLE="UNAVAILABLE"
@dataclass(frozen=True)
class Capability:
    status:str
    code:str
    message_ru:str=""
    full_classical_green_lagrange:bool=False
    @property
    def can_run(self):return self.status in (SUPPORTED,LIMITED)
    def to_dict(self):return {"status":self.status,"code":self.code,
        "message_ru":self.message_ru,"full_classical_green_lagrange":
        self.full_classical_green_lagrange}
def case_capability(case):
    if str(case.get("solver_backend","")).upper()!="VARIATIONAL_REFERENCE":
        return Capability(UNAVAILABLE,"EXPLICIT_VARIATIONAL_BACKEND_REQUIRED")
    if str(case.get("material_model","")).upper()!="HOOKE":
        return Capability(UNAVAILABLE,"ELASTIC_MATERIAL_ONLY")
    if str(case.get("geometry_model","")).upper()!="GREEN_LAGRANGE":
        return Capability(UNAVAILABLE,"FULL_GREEN_LAGRANGE_REQUIRED")
    if str(case.get("mode","")).upper()!="NONLINEAR_ELASTIC":
        return Capability(UNAVAILABLE,"NONLINEAR_ELASTIC_ONLY")
    method=str(case.get("method","")).upper()
    kind=str(case.get("calculation_kind","")).upper()
    if method not in ("MVI","FDM") or kind not in ("STATIC","DYNAMIC"):
        return Capability(UNAVAILABLE,"UNSUPPORTED_METHOD_OR_TIME_MODE")
    return Capability(LIMITED,"REFERENCE_"+kind+"_"+method,
        "Связанная вариационная упругая G3D/Gz постановка")
