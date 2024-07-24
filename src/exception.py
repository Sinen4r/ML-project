import sys
import logging

def error_msg_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_msg="error occured in Python script name [{0}] line number [{1}], erro message[{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_msg

class CustomerException(Exception):
    def __init__(self,error_msg,error_details:sys):
        super().__init__(error_msg)
        self.error_msg=error_msg_detail(error_msg,error_detail=error_details)
    def __str__(self):
        return self.error_msg

# if __name__=="__main__":

#     try :
#         a=1/0
#     except Exception as e:
#         logging.info("divide par zero erro")
#         raise CustomerException(e,sys)