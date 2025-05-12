import subprocess
import webbrowser
import time
import os

print("Launching all servers...")

# Start signaling server
subprocess.Popen(["python", "signaling_server.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)

# Start prediction API
subprocess.Popen(["uvicorn", "sign_api:app", "--host", "0.0.0.0", "--port", "8000"], creationflags=subprocess.CREATE_NEW_CONSOLE)

# Start static HTML server
subprocess.Popen(["python", "-m", "http.server", "8080"], creationflags=subprocess.CREATE_NEW_CONSOLE)

# Wait a bit and then open the page
time.sleep(3)
webbrowser.open("http://localhost:8080/index_with_ip.html")
