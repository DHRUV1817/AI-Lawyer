"""
Redirect script for backward compatibility.
This script redirects users from front-ui.py to app.py for backward compatibility.
"""
import os
import sys
import subprocess

print("⚠️ 'front-ui.py' is deprecated. Please use 'app.py' instead.")
print("Redirecting to app.py...")

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(current_dir, "app.py")

# Check if app.py exists
if not os.path.exists(app_path):
    print("Error: app.py not found in the current directory.")
    sys.exit(1)

# Pass any command line arguments to app.py
args = [sys.executable, app_path] + sys.argv[1:]

# Execute app.py with the same arguments
subprocess.run(args)