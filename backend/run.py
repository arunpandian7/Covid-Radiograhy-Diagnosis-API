import uvicorn
import sys

sys.path.append("../.")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)