<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:01:25 2020

@author: zhong
"""
import numpy as np
from evaluator import evaluate
from PIL import Image

#path of test image
path="./test_doodle/8.jpg"
img = Image.open( path )
data = np.asarray(img)
label=0

#inpute data should be 60*60, label is the class
#'triangle', 'star', 'square', 'circle' representing 0,1,2,3
print(evaluate(data,0))
=======
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:01:25 2020

@author: zhong
"""
import numpy as np
from evaluator import evaluate
from PIL import Image

#path of test image
path="./test_doodle/8.jpg"
img = Image.open( path )
data = np.asarray(img)
label=0

#inpute data should be 60*60, label is the class
#'triangle', 'star', 'square', 'circle' representing 0,1,2,3
print(evaluate(data,0))
>>>>>>> 32daa7e4542856136a20d172f98e99aab85f2650
