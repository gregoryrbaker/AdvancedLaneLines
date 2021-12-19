#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""
==================
Animated histogram
==================

Use a path patch to draw a bunch of rectangles for an animated histogram.
"""
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
from IPython.display import HTML

# Fixing random state for reproducibility
np.random.seed(19680801)

# histogram our data with numpy
data = np.random.randn(1000)
n, bins = np.histogram(data, 100)

# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n
nrects = len(left)

###############################################################################
# Here comes the tricky part -- we have to set up the vertex and path codes
# arrays using ``plt.Path.MOVETO``, ``plt.Path.LINETO`` and
# ``plt.Path.CLOSEPOLY`` for each rect.
#
# * We need 1 ``MOVETO`` per rectangle, which sets the initial point.
# * We need 3 ``LINETO``'s, which tell Matplotlib to draw lines from
#   vertex 1 to vertex 2, v2 to v3, and v3 to v4.
# * We then need one ``CLOSEPOLY`` which tells Matplotlib to draw a line from
#   the v4 to our initial vertex (the ``MOVETO`` vertex), in order to close the
#   polygon.
#
# .. note::
#
#   The vertex for ``CLOSEPOLY`` is ignored, but we still need a placeholder
#   in the ``verts`` array to keep the codes aligned with the vertices.
nverts = nrects * (1 + 3 + 1)
verts = np.zeros((nverts, 2))
codes = np.ones(nverts, int) * path.Path.LINETO
codes[0::5] = path.Path.MOVETO
codes[4::5] = path.Path.CLOSEPOLY
verts[0::5, 0] = left
verts[0::5, 1] = bottom
verts[1::5, 0] = left
verts[1::5, 1] = top
verts[2::5, 0] = right
verts[2::5, 1] = top
verts[3::5, 0] = right
verts[3::5, 1] = bottom

###############################################################################
# To animate the histogram, we need an ``animate`` function, which generates
# a random set of numbers and updates the locations of the vertices for the
# histogram (in this case, only the heights of each rectangle). ``patch`` will
# eventually be a ``Patch`` object.
patch = None


def animate(i):
    # simulate new data coming in
    data = np.random.randn(1000)
    n, bins = np.histogram(data, 100)
    top = bottom + n
    verts[1::5, 1] = top
    verts[2::5, 1] = top
    return [patch, ]

###############################################################################
# And now we build the `Path` and `Patch` instances for the histogram using
# our vertices and codes. We add the patch to the `Axes` instance, and setup
# the `FuncAnimation` with our animate function.
fig, ax = plt.subplots()
barpath = path.Path(verts, codes)
patch = patches.PathPatch(
    barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
ax.add_patch(patch)

ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())

ani = animation.FuncAnimation(fig, animate, 100, repeat=False, blit=True)
plt.show()


# In[6]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport matplotlib.pyplot as plt\nimport matplotlib.animation\nplt.rcParams["animation.html"] = "jshtml"\nimport numpy as np\n\nt = np.linspace(0,2*np.pi)\nx = np.sin(t)\n\nfig, ax = plt.subplots()\nh = ax.axis([0,2*np.pi,-1,1])\nl, = ax.plot([],[])\n\ndef animate(i):\n    l.set_data(t[:i], x[:i])\n\nani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(t))')


# ## Provide Means to Share LaneLine Objects with Pipeline Function

# In[7]:


def proc_vid(op, args):
    print("In proc_vid, calling op with result: ", op(*args))


# In[8]:


class history:
    def __init__(self):
        self.hist = "Blue Cheese"


# In[9]:


class pipeline:
    def __init__(self):
        self.left = history()
        self.right = history()
    
    def __call__(self,x):
        print(self.left.hist)
        print(self.right.hist)
        return x*2


# In[10]:


pl = pipeline()
proc_vid(pl, [3])


# ## Binding Functions to a Class

# In[11]:


def run(self):
    print( self.name, "is running...")
    
class Act:
    def __init__(self, name="Anon"):
        self.name = name

a = Act("Greg")

Act.run = run

a.run()


# ## Calling a free function from a class method

# In[12]:


def run(name='Greg'):
    print( name, "is running...")
    
class Act:
    def __init__(self, name="Anon"):
        self.name = name
        
    def __call__(self):
        run()
        
a = Act()
a()


# ## BiDict
# Ways to use a bi-directional dictionary

# In[13]:


from bidict import bidict


# In[14]:


e = bidict({'A': 1, 'B': 2, 'C': 3, 'D': 4})
e


# In[15]:


e['A']


# In[16]:


e.inverse[4]


# In[17]:


e.keys()


# In[18]:


e.values()


# In[19]:


e._update_no_dup_check({'E': 1})
e.values()


# ## Verify an Instance Passed Via \__call\__ Method Has Access to Other Methods

# In[20]:


from random import uniform


# In[21]:


def first_func(x, history=None):
    print("first_func called with: ", x)
    if history is not None:
        history.coeff(x)
        print("   history.coeff= ", history.coeff())
        return history.coeff()
    else:
        return "No history"
    
    
def second_func(history=None):
    print("second_func called")


# In[22]:


def calling_func(func):
    return func(uniform(-10.0, 10.0))


# In[23]:


class History:
    def __init__(self):
        self._coeff = [0.0]
        
    def coeff(self, val=None):
        if val is None:
            return self._coeff
        else:
            self._coeff.append(val)


# In[24]:


h = History()
h.coeff(1.0)


# In[25]:


class CallableClass:
    def __init__(self):
        self.hasHistory = True
        self.history = History()
    
    def __call__(self, arg):
        return first_func(arg, self.history)


# In[26]:


myInstance = CallableClass()


# In[27]:


calling_func(myInstance)


# In[28]:


myInstance.history.coeff()


# ## Run jupyter nbconvert in pre-commit script

# In[31]:


import sys, os, re
from subprocess import check_output

# Get a list of jupyter notebooks which have been staged for commit 

changed_files = check_output(['git', 'diff', '--cached', '--name-only'])
#changed_files = check_output(['git', 'diff', '--cached', '--stat'])
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


# In[30]:


import re
files = ['a.ipynb', 'a.py', 'b.ipynb', 'b.py', '.git']
r = re.compile(r'.*\.ipynb$')
files = list(filter(r.match, files))
print(files)


# In[ ]:




