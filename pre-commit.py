#!/usr/bin/env python

import sys, os, re
from subprocess import check_output

# Get a list of jupyter notebooks which have been staged for commit 

path = os.getcwd()

print(path)

changed_files = check_output(['git', 'diff', '--cached', '--name-only'])
changed_files = changed_files.decode("utf-8").split("\n")  #split bytes on newlines
changed_files = list(filter(None, changed_files)) #drop empty items

r = re.compile(r".*\.ipynb$")  #regex to match jupyter notebook files

changed_files = list(filter(r.match, changed_files))

changed_files = set(changed_files)
for f in changed_files:
    print(f)
  
#This section uses 'git ls-files' as an alternate method of finding changed files
"""
added_files = check_output(['git', 'ls-files', '--modified', '--deleted', '*.ipynb'])
added_files = added_files.decode("utf-8").split("\n")
added_files = list(filter(None, added_files))
added_files = set(added_files)
print("Added files: ", added_files)
"""
#nb_files = added_files.intersection(changed_files)
nb_files = changed_files

#print("Files = ", nb_files)

# For each notebook, run "jupyter nbconvert --to python <notebook name>"
for f in nb_files:
    results = check_output(['jupyter', 'nbconvert', '--to', 'python', f])
    
# Make a set of files names for the python scripts just created by nbconvert
py_files = set()
for f in nb_files:
    py_files.add(f[:-5]+'py')
    
print("git add the following files: ", py_files)

# Run "git add <notebookname>.py for each notebook converted
for f in py_files:
    results = check_output(['git', 'add', f])
