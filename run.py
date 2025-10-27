import sys
import subprocess

def run_fastapi():
    # Example: running FastAPI with uvicorn
    subprocess.run(["uvicorn", "main:app", "--reload"])

def run_streamlit():
    # Example: running Streamlit
    subprocess.run(["streamlit", "run", "next_ui_openai.py"])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py [fast|streamlit]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "fast":
        run_fastapi()
    elif mode == "streamlit":
        run_streamlit()
    else:
        print("Invalid argument. Use 'fast' or 'streamlit'.")
        sys.exit(1)