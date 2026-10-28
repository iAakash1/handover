import sys, runpy
sys.path.insert(0, r"C:/Users/Admin/AppData/Local/Temp/claude/C--Users-Admin-Desktop-reports/2674d687-2292-479f-ba90-63d322a030b8/scratchpad/tf4")
import transformers; print("transformers", transformers.__version__, flush=True)
sys.argv = ["vlm_eval.py"] + sys.argv[1:]
runpy.run_path(r"C:/Users/Admin/AppData/Local/Temp/claude/C--Users-Admin-Desktop-reports/2674d687-2292-479f-ba90-63d322a030b8/scratchpad/evals/vlm_eval.py", run_name="__main__")
