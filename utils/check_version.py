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


def id_package_version(package_name):
    
    """ This function takes in a package name and returns the version of the package
    Args:
        package_name ([str]): [name of the package you want to check the version of]
    Returns:
        [str]: [version of the package]
    """
    
    assert package_name not in [None,''], 'packge name invalid'
    
    try:
        print(__import__(package_name).__version__)
        package_version = __import__(package_name).__version__
    except :
        package_version= 'Unknown'  
    return package_version

def get_package_versions(file_path):

    """This function takes in a file path and returns a dictionary of package versions used in the file
    Args:
        file_path ([str]): [path to the file you want to check for package versions]
    Returns:
        [dict]: [dictionary of package versions used in the file]
    """

    package_versions = {}

    #open the file to search
    with open(file_path, 'r') as file:

        #read file content to buffer
        content = file.read()

        #id for imports using regex pattern
        matches = re.findall(r'import\s+(\S+)\s*|from\s+(\S+)\s+import', content)

        #loop through the matches
        for match in matches:

            #get the package name from the match
            package_name = match[0] or match[1]

            # if the package name has a '.' in it, then it is a subpackage, and we only need the main package
            if '.' in package_name:
                package_name = package_name.split('.')[0]
            
            #if the package name has a ',' in it, then it is a list of packages, and we need to split it
            if ',' in package_name:

                package_names = package_name.split(',')

                #loop through the package names and add them to the package_versions dictionary if they donot exist already
                for pname in package_names:
                    if pname not in ['',None]:
                        if pname not in package_versions.keys():
                            package_versions[pname] = id_package_version(pname)
            else: 
                    if package_name not in package_versions.keys():
                        package_versions[package_name] = id_package_version(package_name)

    return package_versions

if __name__ =='__main__':
    print(check('pandas'))

    print(get_package_versions('/home/zjc1002/Mounts/code/MyModules/genai/ir_benchmarking/main.py'))
