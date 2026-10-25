"""Discrete Newmark compatibility of moving mechanical Dirichlet traces."""
from __future__ import annotations
import numpy as np


def mechanical_trace_matrix(rows, count):
    selected=[r for r in rows if r.field=="mechanical"]
    c=np.zeros((len(selected),count))
    d=np.zeros(len(selected))
    for i,row in enumerate(selected):
        for j,w in zip(row.indices,row.weights):
            if not 0<=j<count:raise ValueError("mechanical BC trace index out of range")
            c[i,j]+=w
        d[i]=row.prescribed
    norm=np.linalg.norm(c,axis=1)
    if np.any(norm<=0):raise ValueError("empty mechanical BC trace")
    return c/np.where(norm>0,norm,1)[:,None],d/np.where(norm>0,norm,1)


class MechanicalTraceNewmark:
    """Track Cq=d, Cqdot=v, Cqddot=a for fixed C and physical Newmark dt.

    Target velocity and acceleration are determined by Newmark recurrence,
    not substituted with continuous derivatives of the prescribed law.
    State is committed only when all three target residuals pass.
    """
    def __init__(self,c,d,q,v,a,*,t,dt,beta,gamma,atol=1e-10,rtol=1e-8):
        self.c=np.asarray(c,float).copy()
        self.d=np.asarray(d,float).copy()
        self.v=self.c@np.asarray(v,float).ravel()
        self.a=self.c@np.asarray(a,float).ravel()
        self.t,self.dt,self.beta,self.gamma=map(float,(t,dt,beta,gamma))
        self.atol,self.rtol=atol,rtol
        self._verify("initial displacement",self.c@np.asarray(q,float).ravel(),self.d)

    def _verify(self,name,actual,want):
        err=np.abs(actual-want)
        tol=self.atol+self.rtol*np.maximum(np.abs(actual),np.abs(want))
        if np.any(~np.isfinite(err)) or np.any(err>tol):
            raise ValueError(f"Newmark mechanical BC {name} drift: {float(np.max(err,initial=0)):.3e}")
        return float(np.max(err,initial=0))

    def preview(self,c,d,t):
        c=np.asarray(c,float);d=np.asarray(d,float)
        if c.shape!=self.c.shape or np.max(abs(c-self.c),initial=0)>1e-12:
            raise ValueError("moving BC constraint matrix changed")
        if d.shape!=self.d.shape:raise ValueError("moving BC target shape changed")
        if not np.isclose(t,self.t+self.dt,atol=1e-12*self.dt,rtol=1e-12):
            raise ValueError("moving BC time step changed")
        pred=self.d+self.dt*self.v+self.dt**2*(.5-self.beta)*self.a
        accel=(d-pred)/(self.beta*self.dt**2)
        vel=self.v+self.dt*((1-self.gamma)*self.a+self.gamma*accel)
        return (float(t),d.copy(),vel,accel)

    def accept(self,expected,q,v,a):
        time,d,vel,accel=expected
        if not np.isclose(time,self.t+self.dt,atol=1e-12*self.dt,rtol=1e-12):
            raise ValueError("stale moving BC preview")
        defects={
            "displacement":self._verify("displacement",self.c@np.asarray(q,float).ravel(),d),
            "velocity":self._verify("velocity",self.c@np.asarray(v,float).ravel(),vel),
            "acceleration":self._verify("acceleration",self.c@np.asarray(a,float).ravel(),accel),
        }
        self.t,self.d,self.v,self.a=time,d.copy(),vel.copy(),accel.copy()
        return defects
