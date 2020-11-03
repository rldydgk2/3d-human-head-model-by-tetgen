# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:53:41 2020

@author: BMPL
"""

mni=open('./MNI152.mesh','r')
lines=mni.readlines()
l=[]
for i in lines:
    l2=i.split()
    l.append(l2)

data=[]
num=[]    
for i in range(len(l)):
    if len(l[i])==4:
        if int(l[i][3])==1005:
            data.append(l[i])
            num.append(i)

num=num[0]
print(num)

node=[]
face=[]

dic={}
facet={}
for i in data:
    if int(float(i[0]))==float(i[0]) and int(float(i[1]))==float(i[1]) and int(float(i[2]))==float(i[2]):
        face.append(i)
    else :
        node.append(i)

for i in range(len(node)):
    dic[i+1]=[i+1,node[i][0],node[i][1],node[i][2]] #첫뻔째 껀 원래있던 넘버링이다. 

for i in range(len(face)):
    facet[i+1]=[1005,int(face[i][0])-(num+1-5)+1,int(face[i][1])-(num+1-5)+1,int(face[i][2])-(num+1-5)+1]

print(facet)

f=open('./mni2.poly','w')
f.write('# Part 1 - node list\n')
f.write('# node count, 3 dim, no attribute, no boundary marker\n')
f.write('{} 3 0 0\n'.format(len(dic)))
f.write('# Node index, node coordinates\n')
for i in range(len(dic)):
    f.write('{} {} {} {}\n'.format(dic[i+1][0],dic[i+1][1],dic[i+1][2],dic[i+1][3]))
f.write('# Part 2 - facet list\n')
f.write('# facet count, have boundary marker\n')
f.write('{} 1\n'.format(len(facet)))
f.write('# facets\n')
for i in range(len(facet)):
    f.write('1 0 {}\n3 {} {} {}\n'.format(facet[i+1][0],facet[i+1][1],facet[i+1][2],facet[i+1][3]))
f.write('# Part 3 - hole list\n')
f.write('0 # no hole\n')
f.write('# Part 4 - region list\n')
f.write('0 # number of region')
