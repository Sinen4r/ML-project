from setuptools import find_packages,setup
from typing import List
t="-e ."
def get_requirements(file_path:str)->List[str]:
    '''this function will return the list of the requirements'''
    req=[]
    with open(file_path) as file_obj:
        req=file_obj.readlines()
        req=[r.replace("\n","")for r in req]
        if t in req:
            req.remove(t)
    return req
setup(

    name='mlProject',
    version='0.0.1',
    author='Sinen',
    author_email='fradjsinen@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')



)