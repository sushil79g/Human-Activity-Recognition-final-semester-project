import numpy as np
from pprint import pprint

def main(ls):
  sensordata = [ls[i] for i in range(270)] #90(x,y,z)data i.e 270 data 
  inputnode = []
  for a in range(90):
    b = [[sensordata[a*3], sensordata[3*a+1], sensordata[3*a+2]]]
    inputnode.insert(a,b)
  inputnode =[[inputnode]]
  pprint(np.array(inputnode))
