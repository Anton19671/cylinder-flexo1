"""Child process adapter for the restored GUI and the current solver."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from cylinder_paths import activate
activate()
from cylinder_time_integrator_gui import plan_time_integrator_run, execute_time_integrator_plan

def main():
    if len(sys.argv) not in (3,4):
        raise SystemExit('Usage: restored_gui_runner.py CASE INTEGRATOR [--series]')
    if len(sys.argv)==4 and sys.argv[3]!='--series':
        raise SystemExit('Unknown option')
    plan=plan_time_integrator_run(sys.argv[1],sys.argv[2],series=len(sys.argv)==4)
    result=execute_time_integrator_plan(plan)
    print('RESULT:',str(result).encode('ascii','backslashreplace').decode('ascii'))

if __name__=='__main__':
    main()
