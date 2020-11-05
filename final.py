# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:27:48 2020

@author: BMPL
"""

#------------모듈-------------------------------------------+
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
from scipy.linalg import norm
import stl
from stl import mesh
import matplotlib.tri as mtri
import time
import math 

##-----------패러미터--------------------------------------------------+
s=10 #폐곡선 표면부터 극판까지의 거리 
d_1=10 #세라믹 두께
d_2=10 #극판 두께
a=20 #극판의 반지름 
div_c=150 #원판의 둘레를 나눈 수 
now=time.localtime(time.time())
#----------그래프 조정-----------------------------------------------------+
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.xlim(-100,100) #x 축 범위
plt.ylim(-100,100) 
ax.set_zlim(-100,100)

#-------------몸 곡면(구)-----------------------------------------+
def head_surface():
    mni2=open('./mni2.poly','r')
    lines=mni2.readlines()
    l=[]
    for i in lines:
        l2=i.split()
        l.append(l2)

    data=[]
    node={}
    facet={}    
    for i in range(len(l)):
        if not l[i][0]=='#':
            data.append(l[i])
    
    node_num=int(data[0][0])
    
    for i in range(len(data)):
        if 1<=i<=node_num:
            node[i]=[float(data[i][1]),float(data[i][2]),float(data[i][3])]
    face_num=int(data[node_num+1][0])
        
    for i in range(face_num):
        facet[i+1]=[int(data[node_num+2*i+3][1]),int(data[node_num+2*i+3][2]),int(data[node_num+2*i+3][3])]
    
    body_node_num=len(node)
    return node, facet, body_node_num

node, facet, body_node_num=head_surface()
#------------좌표추출-------------------------------------------+
def cor(p):
    if p.count('x')>=1:           
            s=p.split(',')
            b=[]
            for i in range(len(s)):
                b.append(s[i].strip())
            c=[]
            for i in range(len(b)):
                c.append(b[i][2:])
            cx=float(c[0])
            cy=float(c[1])
            cz=float(c[2])
            i=distance(cx,cy,cz,body_node_num)
            kx=node[i][0]
            ky=node[i][1]
            kz=node[i][2]
            return kx,ky,kz,i
    else :
            s=p.split(',')
            b=[]
            for i in range(len(s)):
                b.append(s[i].rstrip('ged '))
            d=[]
            d.insert(0,b[0][8:])
            d.insert(1,b[1][11:])
            print(d)

#-------------거리가까운 메시의 좌표특정을 위한 거 ---------------------+
def distance(x,y,z,limit): #최단거리에 해당하는 node의 넘버를 뽑는것 바디를 구분해야 하기에 limit을 너음   
    d=[]
    for i in range(limit):
        c=(x-node[i+1][0])**2+(y-node[i+1][1])**2+(z-node[i+1][2])**2
        d.append(c)
    e=d.index(min(d))
    return e+1  #+1을 하는 이유는, 현재 노드의 넘버링이 1부터 돼 있기 때문이다. 
'''리턴값을 e로해서, 좌표자체를 특정하는게 아니라 좌표가 리스트에서 존재하는 
위치를 특정하는거임 여기서 e_1은 z의 좌표 e_2 x,y 의 좌표 '''

#-------------------클릭----------------------------------------------+
time = None #클릭 후 시간을 세기위한 장치 
def onclick(event):
    time_interval = 0.25 #0.25초 이내에 더블클릭해야 인식함 
    global time
    if event.button==3: #우클릭시
        p=ax.format_coord(event.xdata,event.ydata) 
        #matplotlib 내장함수. 클릭 위치의 좌표 string으로 추출 
        kx,ky,kz,i=cor(p)
        print(p)
        
        if time is None:
            time = threading.Timer(time_interval, on_singleclick, [event,kx,ky,kz,d_1,d_2,a]) #arg를 튜플형태로 넣어서 싱글클릭에 넣는듯? 
            time.start()
            
        if event.dblclick:
            time.cancel()
            ax.scatter(kx, ky, kz, color='green')
            on_dblclick(event,i,s,d_1,d_2,a)
            

#--------------------MAIN--------------------------------------------+    
##----------- 따블 클릭할 때 ---------------------------------------+
click_num=0
def on_dblclick(event,i,s,d_1,d_2,a):
    global time
    print("You double-clicked", event.button, event.xdata, event.ydata)
    time = None
    flat_plate(i,s,d_1,d_2,a)
    global click_num 
    click_num=click_num+1
    print(click_num)
    return click_num
    
##----------- 싱글 클릭할 때 ---------------------------------------+

def on_singleclick(event,x,y,z,d_1,d_2,a):
    global time
    print("You single-clicked", event.button, event.xdata, event.ydata)
    time = None
    pass

cid = fig.canvas.mpl_connect('button_press_event', onclick)


#-------------노말 단위벡터--------------------------------------+
def n_vector(i): #x,y,z 를 입력하면 벡터가 나오는 것으로 되있는데, 너무 여러번 distance를 돌리는 것 같은데.. 
    l=[]
    for j in list(facet.values()):
        l.append(j[0])
    i2=l.index(i)
    facet_number=i2+1
    p1=np.array(node[facet[facet_number][0]])
    p2=np.array(node[facet[facet_number][1]])
    p3=np.array(node[facet[facet_number][2]])
    v1=p2-p1
    v2=p3-p1
    n_v=np.cross(v1, v2)
    mag = norm(n_v)
    n_v=n_v/mag
    mag2 = norm(v1)
    v1=v1/mag2
    return n_v, v1

#-------------평평한 극판---------------------------------------+
def cylinder(i,s,d_1,d_2,r): 
    '''x,y,z 는 폐곡선 위의 좌표, s는 아랫면 까지의 거리, d_1은 아래쪽 세라믹 두께, d_2는 위쪽 극판두께, r은 반지름'''
    x=node[i][0]
    y=node[i][1]
    z=node[i][2]
    t = np.array([0,d_1,d_1+d_2]) # 층을 3개로 강제로 나눈다
    theta = np.linspace(0, 2 * np.pi, div_c) #2pi 를 50개로 분해 
    radi = np.linspace(0, r, 2) #반지름을 중점과 끝점으로만 분해 
    v, n1=n_vector(i) #법선벡터, 평면단위벡터 1
    
    n2=np.cross(v,n1) 
    mag2=norm(n2)
    n2=n2/mag2 #평면단위벡터 2
    p0=np.array([x,y,z])+s*v # 밑면 중심의 위치 벡터
   

    #use meshgrid to make 2d arrays
    radi,theta1 = np.meshgrid(radi, theta)
    t, theta2 = np.meshgrid(t, theta)

    #generate coordinates for surface
    # "Tube"
    X1, Y1, Z1 = [p0[i] + v[i] * t + r * np.sin(theta2) * n1[i] + r * np.cos(theta2) * n2[i]for i in [0, 1, 2]]
    # "Bottom"
    X2, Y2, Z2 = [p0[i] + radi[i] * np.sin(theta1) * n1[i] + radi[i] * np.cos(theta1) * n2[i]for i in [0, 1, 2]]
    # "Middle"
    X3, Y3, Z3 = [p0[i] + v[i]*d_1 + radi[i] * np.sin(theta1) * n1[i] + radi[i] * np.cos(theta1) * n2[i]for i in [0, 1, 2]]
    # "Top"
    X4, Y4, Z4 = [p0[i] + v[i]*(d_1+d_2) + radi[i] * np.sin(theta1) * n1[i] + radi[i] * np.cos(theta1) * n2[i]for i in [0, 1, 2]]
    

    ax.plot_surface(X1, Y1, Z1, color='red', alpha=0.5)
    ax.plot_surface(X2, Y2, Z2, color='red', alpha=0.5)
    ax.plot_surface(X3, Y3, Z3, color='red', alpha=0.5)
    ax.plot_surface(X4, Y4, Z4, color='red', alpha=0.5)
    return X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4 # 모든좌표를 다 추출한것

#-------------3d flat plate---------------------------------------+
def flat_plate(i,s,d_1,d_2,r):
    '''x,y,z 는 폐곡선 위의 찍은 점의 좌표, s는 찍은 점부터 극판 까지의 거리
    d 는 극판 두께, r은 극판 반지름 '''
    x=node[i][0]
    y=node[i][1]
    z=node[i][2]
    x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4= cylinder(i,s,d_1,d_2,r) #4가 세라믹의 좌표임 
    x_list=pick_p(x2)
    y_list=pick_p(y2)
    z_list=pick_p(z2)
    v=n_vector(i)
    p=[] 
    for i in range(len(x_list)):
        p.append(np.array([x_list[i],y_list[i],z_list[i]]))
    p=np.array(p) # 아랫면의 둘레에 있는 위치벡터들의 리스트 
    
    output2(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4)
    near_point(p)
    num=delit (facet)

    print(len(node))
    print(len(facet))
    export1(node,facet,num)
    
    
#----------보조 함수들 ----------------------------+
def pick_p(a): #원판 둘레의 포인트를 뽑기위한 함수
    b=[]
    for i in range(len(a)):
        b.append(a[i][1])
    b=np.array(b)
    return b

#-------------------------------------------------------------#
def delit (facet):
    many_list=[]
    for i in range(len(facet)):
        many_list.append(i+1)    
    prb=[]  
    for i in range(len(facet)):
        info=[]
        info.append(facet[i+1][0])
        info.append(facet[i+1][1])
        info.append(facet[i+1][2])
        info2=set(info)
        info2=list(info2)
        if not  len(info)==len(info2):
            prb.append(i+1)
            
    a_sub_b = [x for x in many_list if x not in prb]
    return a_sub_b


#-----------아웃풋2,3: 위에 서 구한 실린더의 노드와 폐곡선과 만나는 노드, 페이스들을 모조리 추출한것----+
def output2(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4):
    point=[]

    for i in range(div_c) :
        point.append([x1[i][0],y1[i][0],z1[i][0]])
    for i in range(div_c) : # 합치면 안된다. 노드의 순서가 달라짐 
        point.append([x1[i][1],y1[i][1],z1[i][1]])
    for i in range(div_c) : # 합치면 안된다. 노드의 순서가 달라짐 
        point.append([x1[i][2],y1[i][2],z1[i][2]])
        
    point.append([x2[0][0],y2[0][0],z2[0][0]])
    point.append([x3[0][0],y3[0][0],z3[0][0]])
    point.append([x4[0][0],y4[0][0],z4[0][0]])
    
    many=len(node)
    for i in range(len(point)):
        node[many+i+1]=point[i]
    
    many2=len(facet)
    facet_list=[]
    for i in range(2): #층이 세개여서 면으로 이루어진 층은 세개의 층이다
        for j in range(div_c): 
            if j < div_c-1: #j가 0~48까지, 즉 노드 넘버로는 1~49, 극판의 동그란 면 만드는 것 
                facet_list.append([many+(i)*div_c+j+1,many+(i)*div_c+j+2,many+3*div_c+i+1])
    
    for j in range(div_c): 
            if j < div_c-1: #i=2 일때만 따로 만들어 준 것이다. 바운데리를 나눠야 해서 
                facet_list.append([many+(2)*div_c+j+1,many+(2)*div_c+j+2,many+3*div_c+2+1])
                
            
    for i in range(2): #
        for j in range(div_c): 
                if j < div_c-1: #j가 0~48까지, 즉 노드 넘버로는 1~49, 극판의 옆면만드는 것 
                    facet_list.append([many+i*div_c+j+1,many+i*div_c+j+2,many+(i+1)*div_c+j+2])
                    facet_list.append([many+(i+1)*div_c+j+2,many+(i+1)*div_c+j+1,many+i*div_c+j+1])
            
    for i in range(len(facet_list)):
        facet[many2+i+1]=facet_list[i]

#-----------니어 포인트---------------------------------------------------#
def near_point(p):
    many=len(node)
    many2=len(facet)
    point_num=[] #NODE 넘버의 리스트 
    for i in range(len(p)):
        e=distance(p[i][0],p[i][1],p[i][2],body_node_num)
        point_num.append(e)
    print(point_num)
    for i in range(len(point_num)):
        if i<len(point_num)-1: 
            facet[many2+2*i+1]=[many-(3*div_c+3)+i+1,point_num[i],many-(3*div_c+3)+i+2] 
            facet[many2+2*i+2]=[many-(3*div_c+3)+i+2,point_num[i],point_num[i+1]]
            
    
def export1(node,face,num):
    f=open('./export{}{}{}{}.poly'.format(now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min),'w')
    f.write('# Part 1 - node list\n')
    f.write('# node count, 3 dim, no attribute, no boundary marker\n')
    f.write('{} 3 0 0\n'.format(len(node)))
    f.write('# Node index, node coordinates\n')
    for i in range(len(node)):
        f.write('{} {} {} {}\n'.format(i+1,node[i+1][0],node[i+1][1],node[i+1][2]))
    f.write('# Part 2 - facet list\n')
    f.write('# facet count, have boundary marker\n')
    f.write('{} 1\n'.format(len(num)))
    f.write('# facets\n')
    for i in range(len(num)):
        f.write('1 0 0\n3 {} {} {}\n'.format(facet[num[i]][0],facet[num[i]][1],facet[num[i]][2]))
    
    f.write('# Part 3 - hole list\n')
    f.write('0 # no hole\n')
    f.write('# Part 4 - region list\n')
    f.write('{} # number of region\n'.format(0))

print(len(node))
print(len(facet))