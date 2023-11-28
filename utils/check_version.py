import subprocess
import sys

def check(name):
    """[summary]

    Args:
        name ([str]): [string name of package you want to verify the verison is the latest available]

    Returns:
        [bool]: [True if package is outdated, False if package is bleeding edge ]
    """
    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'list','--outdated'])
    outdated_packages = [r.decode().split('==')[0] for r in reqs.split()]
    return name in outdated_packages


if __name__ =='__main__':
    print(check('pandas'))
