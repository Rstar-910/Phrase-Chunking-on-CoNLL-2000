"""Write pipeline output to a UTF-8 log file for verification."""
import sys
import io

# Redirect all stdout to a UTF-8 file
log = open("pipeline_output.log", "w", encoding="utf-8")
sys.stdout = log
sys.stderr = log

try:
    from main import main
    main()
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    log.close()
