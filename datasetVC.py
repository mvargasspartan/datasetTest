import sys
import platform
from modules.dvc_handler.dvc_handler import handle_Windows, handle_Linux



yourOS = platform.system()

if yourOS == "Windows":
    # Windows stuff
    handle_Windows(sys.argv)
    
elif yourOS == "Linux":
    # Linux stuff
    handle_Linux(sys.argv)




