"""Authoritative GUI descriptors derived from the variational boundary assembler."""
from cylinder_variational_boundary import _fixed_components, HIGHER_TRACES

def theory_from_case(case):
    t=str(case.get("theory","")).upper()
    if t=="CUF":t="CUF_"+str(case.get("cuf_model","")).upper()
    if t not in ("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"):
        raise ValueError("Unknown shell theory")
    return t

def boundary_ui_model(case):
    theory=theory_from_case(case)
    if theory=="KL":
        names=("u0","v0","w0")
    elif theory in ("FSDT","REDDY"):
        names=("u0","v0","w0","theta_x","theta_s")
    else:
        m=int(theory[-1])+1
        names=tuple(f"{component}{k}" for component in ("u","v","w") for k in range(m))
    enabled=(str(case.get("scale_model","")).upper()=="MICRO"
             and bool(case.get("micro",{}).get("enable_flexoelectric")))
    higher=tuple(x for x in HIGHER_TRACES
                 if x!="FIX_FIRST_AND_SECOND_NORMAL" or not theory.startswith("CUF_"))
    support_names={}
    for name in ("FREE","CLAMPED","SIMPLY_SUPPORTED","GUIDED","SYMMETRY"):
        support_names[name]=tuple(names[i] for i in _fixed_components(name,theory,len(names),{}))
    return {"theory":theory,"components":names,"n_components":len(names),
            "support_components":support_names,
            "higher_choices":higher,"electric_enabled":enabled,
            "electric_sites":("left","right","outer","inner") if enabled else (),
            "electric_modes":("INSULATED","POTENTIAL","CHARGE") if enabled else ("INSULATED",)}

__all__=("theory_from_case","boundary_ui_model",)
