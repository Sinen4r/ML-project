import logging
import os 
from datetime import datetime

log_file=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",log_file)
os.makedirs(logs_path,exist_ok=True)

LogFilePath=os.path.join(logs_path,log_file)

logging.basicConfig(
    filename=LogFilePath,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# if __name__=="__main__":
#     logging.info("logging has started")