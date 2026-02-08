import subprocess
import os
import sys
import time
import threading

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")

def check_calibration():
    conv = os.path.join(DATA_DIR, "calibration_convergent.csv")
    div = os.path.join(DATA_DIR, "calibration_divergent.csv")
    return os.path.exists(conv) and os.path.exists(div)

def stream_logs(process, prefix):
    """Prints internal logs and errors from the scripts."""
    for line in iter(process.stderr.readline, ''):
        if line: print(f"❌ {prefix} ERROR: {line.strip()}")

def main():
    print("\n" + "="*50 + "\n   🧠 NEURO-LINK COMMAND CENTER\n" + "="*50)
    
    if not check_calibration():
        print("[!] ERROR: Calibration files not found. Run src/calibration.py first.")
        return
            
    print("\n   🚀 INITIALIZING SYSTEM...")

    processes = []
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_DIR

    try:
        # Start Watcher
        p_watcher = subprocess.Popen(
            [sys.executable, "watcher.py"], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=SRC_DIR, env=env
        )
        processes.append(p_watcher)
        threading.Thread(target=stream_logs, args=(p_watcher, "WATCHER"), daemon=True).start()

        time.sleep(3) # Vital: Give the CSV file time to be created

        # Start Analyzer
        p_analyzer = subprocess.Popen(
            [sys.executable, "analyzer.py"], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=SRC_DIR, env=env
        )
        processes.append(p_analyzer)
        threading.Thread(target=stream_logs, args=(p_analyzer, "ANALYZER"), daemon=True).start()

        print("✅ LIVE ANALYTICS STREAMING:\n")
        
        while True:
            line = p_analyzer.stdout.readline()
            if line:
                print(f"📊 {line.strip()}")
            
            # Check if processes are alive
            if p_watcher.poll() is not None:
                print("\n[ALERT] Watcher terminated.")
                break
            if p_analyzer.poll() is not None:
                print("\n[ALERT] Analyzer terminated.")
                break

    except KeyboardInterrupt:
        print("\n🛑 SHUTTING DOWN...")
    finally:
        for p in processes: p.terminate()
        print("Done.")

if __name__ == "__main__":
    main()