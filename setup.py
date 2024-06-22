from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function returns the list of requirements
    '''
    Requirements=[]
    with open(file_path) as file_obj:
        Requirements=file_obj.readlines()
        Requirements=[req.replace("\n","") for req in Requirements]
        
        if HYPHEN_E_DOT in Requirements:
            Requirements.remove(HYPHEN_E_DOT)
    return Requirements

setup(
name='MLproject',
version='0.0.1',
author='Annas',
author_email='annaskhalidkhan@gmail.com',
packages=find_packages(),
install_requirements=get_requirements('requirements.txt')
)