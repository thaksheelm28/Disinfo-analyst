"""
server/app.py — FastAPI server for Disinformation Analyst (OpenEnv)
Entry point: main() called via [project.scripts] server = "server.app:main"
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import app   # re-export the same FastAPI app from root app.py

def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Server entry point — called by [project.scripts] server = 'server.app:main'"""
    import uvicorn
    uvicorn.run("server.app:app", host=host, port=port, workers=1)

if __name__ == "__main__":
    main()
