
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.tri as tri
from tqdm import tqdm
##-----------패러미터--------------------------------------------------+
switch='all' #1개 붙일지 모두 붙일지 선택하는 변수 


s=3.0 #폐곡선 표면부터 극판까지의 거리, 하이드로 겔 의 두께임 (mm 단위임 )
d_1=0.3 #세라믹 두께
d_2=1.0 #극판 두께
a=9.0 #극판의 반지름 
div_c=10 #원판의 둘레를 나눈 수 
edge_xy=24 # 사각형 패치의 xy 방향 극판 중심 사이의 거리 
edge_z=30 # 사각형 패치의 z축 방향 극판 중심 사이의 거리 
div_t=30 # 종양의 둘레를 나눈 수 
div_tz=30 # 종양의 z축 방향을 나눈 수 
r_1=5.0 # 종양 겉면 반지름 
r_2=0.8*r_1 # 종양 속 반지름 
center_x, center_y, center_z = 33, -45, 23 #종양 위치 
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

#-------------헤드 모델 -----------------------------------------+
def head_surface():
    mni2=open('./mni4.poly','r')
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
            node[i]=[int(data[i][4]),float(data[i][1]),float(data[i][2]),float(data[i][3])]
    face_num=int(data[node_num+1][0])
        
    for i in range(face_num):
        facet[i+1]=[int(data[node_num+2*i+2][2]),int(data[node_num+2*i+3][1]),int(data[node_num+2*i+3][2]),int(data[node_num+2*i+3][3])]
    
    region={}
    
    skin={}
    for i in range(len(node)):
        if node[i+1][0]==1005:
           skin[i+1]=np.array(node[i+1][1:]) #앞에는 node 에서의 넘버링, 뒤에는 좌표, 이거는 일부러 어레이로 만듬 
    
    skin_facet={}
    for i in facet:
        if facet[i][0]==1005:
            skin_facet[i]=facet[i][1:]
            
    return node, facet, region, skin, skin_facet


node, facet, region, skin, skin_facet =head_surface()
dic={}

#------------종양 모델 ------------------------------+
def tumor_surface (radius, center_x, center_y, center_z): #반지름, 구의 중심 좌표 
    theta = np.linspace(0, 2 * np.pi, div_t)
    z = np.linspace(center_z-radius, center_z+radius, div_tz)
    radi=[]
    z_array=[]
    for i in range(len(z)):
        r=(radius**2-abs(z[i]-center_z)**2)**0.5
        radi.append(r)
        z_array.append(np.array([z[i]]))
    
    radi=np.array(radi)
    z_array=np.array(z_array)
    x = np.outer(radi, np.cos(theta))+center_x
    y = np.outer(radi, np.sin(theta))+center_y
    ax.plot_surface(x, y, z_array, color='yellow',alpha=0.3)
    return x,y,z_array

xi, yi, zi = tumor_surface(r_1,center_x, center_y, center_z)
xi2,yi2,zi2 = tumor_surface(r_2,center_x, center_y, center_z)


def add_tumor(xi,yi,zi,boundary_num):
    many_node=max(node.keys())
    point=[]
    for i in range(div_tz):
        if i==0:
            point.append([2,xi[i][0],yi[i][0],zi[i][0]]) # 2는 임의로 정한 종양 노드의 넘버링 
            
        elif 0<i<div_tz-1:
            for j in range(div_t-1):
                point.append([2,xi[i][j],yi[i][j],zi[i][0]])
        
        elif i==div_tz-1:
            point.append([2,xi[i][0],yi[i][0],zi[i][0]])
    
    for i in range(len(point)):
        node[many_node+i+1]=point[i]
             
    facet_list=[]
    
    for i in range(div_tz-1):# 순서대로 바운더리 마커, 노드수, 노드 넘버 
        if i==0:
            for j in range(div_t-1):
                if j < div_t-2: #j가 0~98까지, 즉 노드 넘버로는 1~99
                    facet_list.append([boundary_num,many_node+1,many_node+j+2,many_node+j+3]) #맨앞에 4은 바운데리 마커, 3은 노드 수  
                else : 
                    facet_list.append([boundary_num,many_node+1,many_node+2,many_node+j+2])
                    
        elif 0<i<div_tz-2 :
            for j in range(div_t-1):
                if j < div_t-2: #j가 0~98까지, 즉 노드 넘버로는 1~99
                    facet_list.append([boundary_num,many_node+(div_t-1)*(i-1)+j+2,many_node+(div_t-1)*(i-1)+j+3,many_node+(div_t-1)*(i)+j+3])
                    facet_list.append([boundary_num,many_node+(div_t-1)*(i)+j+3,many_node+(div_t-1)*(i)+j+2,many_node+(div_t-1)*(i-1)+j+2])
                else : 
                    facet_list.append([boundary_num,many_node+(div_t-1)*(i-1)+j+2,many_node+(div_t-1)*(i-1)+0+2,many_node+(div_t-1)*(i)+0+2])
                    facet_list.append([boundary_num,many_node+(div_t-1)*(i)+0+2,many_node+(div_t-1)*(i)+j+2,many_node+(div_t-1)*(i-1)+j+2])
            
        elif i==div_tz-2:
            for j in range(div_t-1):
                if j < div_t-2: #j가 0~98까지, 즉 노드 넘버로는 1~99
                    facet_list.append([boundary_num,many_node+(div_t-1)*(i-1)+j+2,many_node+(div_t-1)*(i-1)+j+3,many_node+(div_t-1)*(i)+0+2])
                else :    
                    facet_list.append([boundary_num,many_node+(div_t-1)*(i-1)+j+2,many_node+(div_t-1)*(i-1)+0+2,many_node+(div_t-1)*(i)+0+2])
                       
    many_facet=max(facet.keys())

    for i in range(len(facet_list)):
        facet[many_facet+i+1]=facet_list[i]
    
    region[1000]=[(r_1+r_2)/2+center_x, center_y, center_z,4]
    region[1001]=[center_x, center_y, center_z,5]


add_tumor(xi,yi,zi,1007)
add_tumor(xi2,yi2,zi2,1008)

'''#-----------리전 만드는 함수------------------------------------------------+
def make_region(): #바디와 종양의 리전을 만드는 함수
    
'''

#------------질량줌심을 만드는 함수----------------------------------------------------------+
def make_mass_point_of_facet():
    mass_p_dic={}
    for i in skin_facet:
        mass_p=(skin[skin_facet[i][0]]
        +skin[skin_facet[i][1]]+skin[skin_facet[i][2]])/3
        mass_p_dic[i]=mass_p
    return mass_p_dic

mass_p_dic=make_mass_point_of_facet()

#---------네이버 만드는 함수---------------------------------------------------------------+
def make_neighbor(interest_facet): # 한페이셋의 주변 페이셋들을 가르쳐 주는 딕셔너리 
    neighbor={}    
    for i in tqdm(interest_facet, desc="making neighbor", mininterval=1):
        l={}
        for j in interest_facet:
            if not i==j:
                k=set(interest_facet[i])-set(interest_facet[j])
                if len(k)==1:
                    if interest_facet[i][0]==list(k)[0]:
                        l[0]=j
                    if interest_facet[i][1]==list(k)[0]:
                        l[1]=j
                    if interest_facet[i][2]==list(k)[0]:
                        l[2]=j
        p=[]
        if len(l)==3:
            p.append(l[0])
            p.append(l[1])
            p.append(l[2])
            neighbor[i]=p
        
    return neighbor

#------------ 트라이 앵글 메쉬 그리기(맷플롯 립으로 비주얼 라이즈 할라고 만들었는데, 너무 무거움)----------------------------+
'''def triangle(node,facet):
    element=[]
    for i in range(len(facet)):
        element.append([node[facet[i+1][1]],node[facet[i+1][2]],node[facet[i+1][3]]])
    ax.add_collection3d(Poly3DCollection(element, 
    facecolors='cyan', linewidths=0.1, edgecolors='y', alpha=0.1))
    plt.show()
triangle(node,facet)'''

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
            i=distance(cx,cy,cz,skin)
            kx=skin[i][0]
            ky=skin[i][1]
            kz=skin[i][2]
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

#-------------거리가까운 노드의 넘버특정을 위한 거 ---------------------+
def distance(x,y,z,interesting_node): #최단거리에 해당하는 node의 넘버를 뽑는것 바디를 구분해야 하기에 limit을 너음   
    d=[]
    indexx=[]
    
    for i in interesting_node:
        c=(x-interesting_node[i][0])**2+(y-interesting_node[i][1])**2+(z-interesting_node[i][2])**2
        d.append(c)
        indexx.append(i)
    
    e=d.index(min(d))
    return indexx[e]

#-------------------클릭----------------------------------------------+
time = None #클릭 후 시간을 세기위한 장치 
def onclick(event):
    time_interval = 0.25 #0.25초 이내에 더블클릭해야 인식함 
    global time
    if event.button==3: #우클릭시
        p=ax.format_coord(event.xdata,event.ydata) #matplotlib 내장함수. 클릭 위치의 좌표 string으로 추출 
        kx,ky,kz,i=cor(p)
        print(p)
        
        if time is None:
            time = threading.Timer(time_interval, on_singleclick, [event,kx,ky,kz,d_1,d_2,a]) #arg를 튜플형태로 넣어서 싱글클릭에 넣는듯? 
            time.start()
            
        if event.dblclick:
            time.cancel()
            ax.scatter(kx, ky, kz, color='green')
            on_dblclick(event,switch,i,s,d_1,d_2,a)
            

#--------------------MAIN--------------------------------------------+    
##----------- 따블 클릭할 때 ---------------------------------------+
click_num=0
def on_dblclick(event,switch,i,s,d_1,d_2,a):
    global time
    print("You double-clicked", event.button, event.xdata, event.ydata)
    time = None
    
    if switch==1:
        flat_plate(0,i,s,d_1,d_2,a)
        
    elif switch=='all':    
        all_plate(i,edge_xy, edge_z)
    
    num=delit(facet)
    export1(node,facet,num)
    export2(dic)
    
    global click_num 
    click_num=click_num+1
    print('click_num=%d'%click_num)
    return click_num
    
##----------- 싱글 클릭할 때 ---------------------------------------+

def on_singleclick(event,x,y,z,d_1,d_2,a):
    global time
    print("You single-clicked", event.button, event.xdata, event.ydata)
    time = None
    pass

cid = fig.canvas.mpl_connect('button_press_event', onclick)


#-------------노말 단위벡터--------------------------------------+
def n_vector(i):  #여기서 i 는 노드의 넘버링 
    for j in skin_facet:
        if i==skin_facet[j][0]:
            break
        if i==skin_facet[j][1]:
            break
        if i==skin_facet[j][2]:
            break
    facet_number=j
    p1=skin[skin_facet[facet_number][0]]
    p2=skin[skin_facet[facet_number][1]]
    p3=skin[skin_facet[facet_number][2]]
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
    x=skin[i][0]
    y=skin[i][1]
    z=skin[i][2]
    t = np.array([0,d_1,d_1+d_2]) # 층을 3개로 강제로 나눈다
    theta = np.linspace(0, 2 * np.pi, div_c) #2pi 를 div_c개로 분해 
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
def flat_plate(j,i,s,d_1,d_2,r):
    '''x,y,z 는 폐곡선 위의 찍은 점의 좌표, s는 찍은 점부터 극판 까지의 거리
    d 는 극판 두께, r은 극판 반지름 '''
    interest_facet=make_interesting_facet(i,detec_l=15)
    #interest_skin=make_interesting_skin(i,detec_l=15)
    x=skin[i][0]
    y=skin[i][1]
    z=skin[i][2]
    x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4= cylinder(i,s,d_1,d_2,r) #4가 세라믹의 좌표임 
    x_list=pick_p(x2)
    y_list=pick_p(y2)
    z_list=pick_p(z2)
    v,n1=n_vector(i) # v 가 실린더 관통하는 노말 벡터 
    p=[] 
    for i in range(len(x_list)):
        p.append(np.array([x_list[i],y_list[i],z_list[i]]))
    p=np.array(p) # 아랫면의 둘레에 있는 위치벡터들의 리스트 
    
    new_node_in_circle=output_node_facet(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4)
    output_region(j,x,y,z,v)
    #near_point(p,interest_skin)
    
    
    neighbor=make_neighbor(interest_facet)
    
    many_dic=len(dic)
    make_hydrogel(interest_facet,neighbor,new_node_in_circle,v,many_dic) 
    
    print(len(node))
    print(len(facet))
    
    
#----------보조 함수 ----------------------------+
def pick_p(a): #원판 둘레의 포인트를 뽑기위한 함수
    b=[]
    for i in range(len(a)):
        b.append(a[i][1])
    b=np.array(b)
    return b

#-----------아웃풋2,3: 위에 서 구한 실린더의 노드와 폐곡선과 만나는 노드, 페이스들을 모조리 출력하여 node, facet 딕셔너리에 넣은것 -----------+
def output_node_facet(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4):
    point=[]

    for i in range(div_c-1) :
        point.append([x1[i][0],y1[i][0],z1[i][0]])
    for i in range(div_c-1) : # 합치면 안된다. 노드의 순서가 달라짐 
        point.append([x1[i][1],y1[i][1],z1[i][1]])
    for i in range(div_c-1) : # 합치면 안된다. 노드의 순서가 달라짐 
        point.append([x1[i][2],y1[i][2],z1[i][2]])
        
    point.append([x2[0][0],y2[0][0],z2[0][0]])
    point.append([x3[0][0],y3[0][0],z3[0][0]])
    point.append([x4[0][0],y4[0][0],z4[0][0]])
    
    many=max(node.keys())
    for i in range(len(point)):
        node[many+i+1]=[1]+point[i] # 여기서 1 이 붙은 이유는 마커를 붙인 것이다. 

    many2=max(facet.keys())
    facet_list=[]
    for i in range(2): #층이 세개여서 면으로 이루어진 층은 세개의 층이다
        for j in range(div_c-1): #j가 0~48까지, 즉 노드 넘버로는 1~49, 극판의 동그란 면 만드는 것 
            if j<div_c-2: 
                facet_list.append([-1,many+(i)*(div_c-1)+j+1,many+(i)*(div_c-1)+j+2,many+3*(div_c-1)+i+1])
            else:
                facet_list.append([-1,many+(i)*(div_c-1)+j+1,many+(i)*(div_c-1)+0+1,many+3*(div_c-1)+i+1])
    
    for j in range(div_c-1): #i=2 일때만 따로 만들어 준 것이다. 전압주는 부분의 바운데리를 나눠야 해서     
        if j<div_c-2:     
            facet_list.append([click_num+1,many+(2)*(div_c-1)+j+1,many+(2)*(div_c-1)+j+2,many+3*(div_c-1)+2+1])
        else:
            facet_list.append([click_num+1,many+(2)*(div_c-1)+j+1,many+(2)*(div_c-1)+0+1,many+3*(div_c-1)+2+1])
                
            
    for i in range(2): #
        for j in range(div_c-1): #j가 0~48까지, 즉 노드 넘버로는 1~49, 극판의 옆면만드는 것
                if j < div_c-2:  #j=0~47
                    facet_list.append([-1,many+i*(div_c-1)+j+1,many+i*(div_c-1)+j+2,many+(i+1)*(div_c-1)+j+2])
                    facet_list.append([-1,many+(i+1)*(div_c-1)+j+2,many+(i+1)*(div_c-1)+j+1,many+i*(div_c-1)+j+1])
                else: #48
                    facet_list.append([-1,many+i*(div_c-1)+j+1,many+i*(div_c-1)+0+1,many+(i+1)*(div_c-1)+0+1])
                    facet_list.append([-1,many+(i+1)*(div_c-1)+0+1,many+(i+1)*(div_c-1)+j+1,many+i*(div_c-1)+j+1])
            
    for i in range(len(facet_list)):
        facet[many2+i+1]=facet_list[i]
    
    new_node_in_circle=range(many+1,many+div_c)
    return new_node_in_circle

def output_region(j,x,y,z,v): # 극판의 리전을 만드는 것 
    many_region=len(region)
    point=np.array([x,y,z])
    point1=point+(s/2)*v
    point2=point+(s+d_1/2)*v
    point3=point+(s+d_1+d_2/2)*v
    region[many_region+1]=[point1[0],point1[1],point1[2],1] # 리전 1 은 하이드로 갤
    region[many_region+2]=[point2[0],point2[1],point2[2],2] # 리전 2 은 세라믹 
    region[many_region+3]=[point3[0],point3[1],point3[2],3] # 리전 3 은 구리

#----------- 니어 포인트, 하이드로겔 부분 만드는 것 ---------------------------------------------------+
def near_point(p,interest_skin):
    many=max(node.keys())
    many2=max(facet.keys())
    point_num=[] #NODE 넘버의 리스트 
    for i in range(len(p)-1):
        e=distance(p[i][0],p[i][1],p[i][2],interest_skin)
        point_num.append(e)
        
    for i in range(len(point_num)): #0~48
        if i<len(point_num)-1: #0~47
            facet[many2+2*i+1]=[-1,many-(3*div_c)+i+1,point_num[i],many-(3*div_c)+i+2] 
            facet[many2+2*i+2]=[-1,many-(3*div_c)+i+2,point_num[i],point_num[i+1]]
        else : #48
            facet[many2+2*i+1]=[-1,many-(3*div_c)+i+1,point_num[i],many-(3*div_c)+0+1] 
            facet[many2+2*i+2]=[-1,many-(3*div_c)+0+1,point_num[i],point_num[0]]
            

#-----------중복된 facet들을 지워주고, 남은 facet들의 넘버링을 알려주는 함수 --------------------------------------------------+
def delit (facet):
    many_list=[]
    for i in facet:
        many_list.append(i)    
    prb=[]  
    for i in facet:
        info=[]
        info.append(facet[i][1])
        info.append(facet[i][2])
        info.append(facet[i][3])
        info2=set(info)
        info2=list(info2)
        if not  len(info)==len(info2):
            prb.append(i)
            
    a_sub_b = [x for x in many_list if x not in prb]
    return a_sub_b


#--------출력 하기 위한 함수 (현재 극판이 18개가 한꺼번에 붙어서 18번 반복되는 문제점이 있음 )------------------------------#
def export1(node,face,num):
    f=open('./export{}{}{}{}.poly'.format(now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min),'w')
    f.write('# Part 1 - node list\n')
    f.write('# node count, 3 dim, no attribute, no boundary marker\n')
    f.write('{} 3 0 0\n'.format(len(node)))
    f.write('# Node index, node coordinates\n')
    for i in range(len(node)):
        f.write('{} {} {} {}\n'.format(i+1,node[i+1][1],node[i+1][2],node[i+1][3]))
        
    f.write('# Part 2 - facet list\n')
    f.write('# facet count, have boundary marker\n')
    f.write('{} 1\n'.format(len(facet)))
    f.write('# facets\n')
    for i in facet:
        if len(facet[i])==4:
            f.write('1 0 {}\n3 {} {} {}\n'.format(facet[i][0],facet[i][1],facet[i][2],facet[i][3]))
        elif len(facet[i])==5:
            f.write('1 0 {}\n4 {} {} {} {}\n'.format(facet[i][0],facet[i][1],facet[i][2],facet[i][3],facet[i][4]))
    f.write('# Part 3 - hole list\n')
    f.write('0 # no hole\n')
    f.write('# Part 4 - region list\n')
    f.write('{} # number of region\n'.format(len(region)))
    for i in region:
        f.write('{} {} {} {} {}\n'.format(i, region[i][0],region[i][1],region[i][2],region[i][3]))
    

#-----------9개의 극판을 한꺼번에 붙여주기 위한 함수---------------------------------------------+
def multi_i(i,edge_xy, edge_z): # 한점 잡았을때 9개의 점을 잡아준는 펑션, i는 노드 넘버,edge는 사각형의 변길이의 절반  
    n_v,v1=n_vector(i) #우선 뽑은 점의 법선벡터를 구함
    #d=np.dot(skin[i],n_v)
    v_1=np.cross(np.array([0,0,1]),n_v) #z축 방향이 아닌 벡터, 즉 xy 벡터, 반시계로 돌게 헀음 # 정수리 
    if norm(v_1)==0:
        v_1=np.cross(np.array([1,0,0]),n_v)
    v_1=v_1/norm(v_1)
    v_2=np.cross(n_v,v_1)
    v_2=v_2/norm(v_2) #z축방향 변화, +쪽이 위쪽임 
    p=[]
    p.append(skin[i]+edge_xy*v_1)
    p.append(skin[i]+edge_xy*v_1+edge_z*v_2)
    p.append(skin[i]+edge_z*v_2)
    p.append(skin[i]+edge_xy*-v_1+edge_z*v_2)
    p.append(skin[i]+edge_xy*-v_1)
    p.append(skin[i]+edge_xy*-v_1-edge_z*v_2)
    p.append(skin[i]+edge_z*-v_2)
    p.append(skin[i]+edge_xy*v_1-edge_z*v_2)
    result=[]
    result.append(i)
    
    interest_skin2=make_interesting_skin(i,100)
    for i in p:
        result.append(distance(i[0],i[1],i[2],interest_skin2))
        
    return result 

    
#-----------클릭한 반대편의 노드 넘버를 알려주는 함수------------------------------------+
def opposit_side(i,n_v):
    point_vec={}
    for j in skin:
        v=skin[j]-skin[i]
        if not norm(v)==0:
            v=v/norm(v)
            point_vec[j]=v
        else:    
            point_vec[j]=v
    
    dot={}
    for j in point_vec:
        if not norm(point_vec[j])==0:
            dot[j]=abs(np.dot(n_v,point_vec[j]))
                
    
    k=list(dot.values()).index(max(dot.values()))
    i2=list(dot.keys())[k]
    i3=distance(skin[i2][0],skin[i2][1],skin[i][2],skin)
    return i3



#-----------총 18개의 극판을 붙이게 해주는 함수------------------------------------+   
def all_plate(i,edge_xy, edge_z):
    result1=multi_i(i,edge_xy, edge_z)
    n_v,v1=n_vector(i)
    i3=opposit_side(i,n_v)
    result2=multi_i(i3,edge_xy, edge_z)
    result=result1+result2
    global click_num
    for j in tqdm(range(len(result)), desc="plate", mininterval=1):
        if j<=8:
            flat_plate(j+9*click_num,result[j],s,d_1,d_2,a)
        elif j==9:
            click_num=click_num+1
            flat_plate(j+9*(click_num-1),result[j],s,d_1,d_2,a)
        else: 
            flat_plate(j+9*(click_num-1),result[j],s,d_1,d_2,a)

    return click_num
    

#----------아랫 극판의 둘레에서 벡터를 가지고 내렸을때 특정한 facet-----------------+
def find_facet(interest_facet,i,n_v):
    
    for j in interest_facet :
        v1=skin[interest_facet[j][0]]
        v2=skin[interest_facet[j][1]]
        v3=skin[interest_facet[j][2]]
        x1=v2-v1
        x2=v3-v2
        x3=v1-v3
        v=np.array(node[i][1:])
        k1=np.dot(n_v,np.cross(x1,(v-v1)))
        k2=np.dot(n_v,np.cross(x2,(v-v2)))        
        k3=np.dot(n_v,np.cross(x3,(v-v3)))
        
        if k1>0:
            k1=1   # 1인 경우가 +, 0인 경우가 - 인걸로 판단한 거다. 
        else :
            k1=0
        
        if k2>0:
            k2=1
        else :
            k2=0
        
        if k3>0:
            k3=1
        else :
            k3=0
            
        if k1==k2==k3==1:
            k=j
            break
            
    
    return k    


#-----------평면과 레이의 크로스 되는 포인트를 만들기 위한 함수---------------------+
def cross_point(i,j,n_v): #여기서 i는 원판위의 점의 노드 넘버, j는 facet의 넘버를 의미한다. 
    '''x=node[i][1]
    y=node[i][2]
    z=node[i][3]'''
    p0=np.array(node[i][1:])
    
    p1=skin[skin_facet[j][0]]
    p2=skin[skin_facet[j][1]]
    p3=skin[skin_facet[j][2]]
    v1=p2-p1
    v2=p3-p1
    v=np.cross(v1, v2) #facet의 노말 벡터 
    '''a=v[0]
    b=v[1]
    c=v[2]
    d=v[0]*p1[0]+v[1]*p1[1]+v[2]*p2[2]
    #h=abs(a*x+b*y+c*z-d)/(a**2+b**2+c**2)**0.5 #평면과 원판위 점 사이의 거리
    #cos=abs(np.dot(np.array([a,b,c]),n_v))/(a**2+b**2+c**2)**0.5
    #l=(a*x+b*y+c*z-d)/(np.dot(np.array([a,b,c]),n_v))#h/cos #점과 페이셋을 이었을때, n_v로 이동했을 때의 거리 '''
    l=(np.dot(v,(p0-p1)))/(np.dot(v,n_v))
    p=p0-l*n_v
    ax.scatter(p[0], p[1], p[2], color='blue')
    return p


#----------노드 출력 함수---------------------------------------------------------+
def export2(dic):
    f=open('./export{}{}{}{}.node'.format(now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min),'w')
    f.write('{}  3  0  0\n'.format(len(dic)))
    for i in range(len(dic)):
        f.write('   {}    {}  {}  {}\n'.format(i+1,dic[i+1][0],dic[i+1][1],dic[i+1][2]))


#---------부호판단--------------------------------------------------------------------+
def determine_direction(neighbor,i,p,p0,n_v) : # i 는 페이셋의 넘버, p는 임의의 한 점, p0는 내부의 한 점 
    p1=skin[skin_facet[i][0]]
    p2=skin[skin_facet[i][1]]
    p3=skin[skin_facet[i][2]]
    v1=p2-p1
    v2=p3-p1
    x1=p2-p1
    x2=p3-p2
    x3=p1-p3
    v=np.cross(v1, v2) #facet의 노말 벡터
    k1=np.dot(n_v,np.cross(x1,(p-p1)))
    k2=np.dot(n_v,np.cross(x2,(p-p2)))
    k3=np.dot(n_v,np.cross(x3,(p-p3)))
    if k1>=0 :
        k1=1
    else :
        k1=-1
    
    if k2>=0 :
        k2=1
    else :
        k2=-1
    
    if k3>=0 :
        k3=1
    else :
        k3=-1
    
    k= [k1,k2,k3]
    if k==[1,-1,1]:
        out=neighbor[i][0]#1번 이웃 페이셋
        case=1
    elif k==[1,1,-1]:
        out=neighbor[i][1]
        case=2
    elif k==[-1,1,1]:
        out=neighbor[i][2]
        case=3
    
    elif k==[1,-1,-1]:
        line=np.dot(n_v,np.cross((p-p0),(p3-p0)))
        if line >=0 :
            out=neighbor[i][0]
            case=1
        else:
            out=neighbor[i][1]
            case=2
    elif k==[-1,1,-1]:
        line=np.dot(n_v,np.cross((p-p0),(p1-p0)))
        if line >=0 :
            out=neighbor[i][1]
            case=2
        else:
            out=neighbor[i][2]
            case=3
    elif k==[-1,-1,1]:
        line=np.dot(n_v,np.cross((p-p0),(p2-p0)))
        if line >=0 :
            out=neighbor[i][2]
            case=3
        else:
            out=neighbor[i][0]
            case=1
    
    elif k==[1, 1,1] or k==[-1,-1,-1] :
        out = i 
        case=0
    
    x=cross_point2(i,out,p,p0,v,case,n_v)
    #print('case=%s'%case)
    return out, case, x

#--------페이셋 모서리와 두점을 포함하는 평면의 교점--------------------------------------+
def cross_point2(i,out,p,p0,v,case,n_v):
    nomal_v=np.cross(p-p0,n_v)
    
    if case==1:
        t1=skin[skin_facet[i][1]]
        t2=skin[skin_facet[i][2]]
        
    elif case==2:
        t1=skin[skin_facet[i][2]]
        t2=skin[skin_facet[i][0]]
    
    elif case==3:
        t1=skin[skin_facet[i][0]]
        t2=skin[skin_facet[i][1]]
    
    elif case==0:
        t1=np.array([0,0,1])
        t2=np.array([0,0,0])
        
    t_v=t1-t2
    
    if case==1 or case==2 or case==3:
        l=(np.dot(nomal_v,(t1-p0)))/(np.dot(nomal_v,t_v))  #h/cos #점과 페이셋을 이었을때, n_v로 이동했을 때의 거리 
        x=t1-l*t_v
        if not 0<=l<=1 :
            print('error')
    elif case==0 :
        l=0
        x=p    
    
    return x

#----------------------------------------------------------------------------+
def make_hydrogel(interest_facet,neighbor,new_node_in_circle,v,many_dic):
    face=[]
    point=[]
    
    for i in tqdm(range(len(new_node_in_circle)), desc="making point", mininterval=1):
        number_facet=find_facet(interest_facet,new_node_in_circle[i],v)
        face.append(number_facet)
        p=cross_point(new_node_in_circle[i],number_facet,v)
        dic[many_dic+1+i]=p
        point.append(p)
        
    
    face_point=[]
    for u in range(len(point)):
        if u == 0:
            p0=point[u]
            p=point[u+1]
            out=face[u]
            l=[out,[],[p0]]
            out1, case, p0=determine_direction(neighbor,out,p,p0,v)
            l[1].append(case)
            l[2].append(p0)
            face_point.append(l)
            
            while not out1==face[u+1]:
                case=find_neighbor(neighbor,out1,out)
                l2=[out1,[case],[p0]]
                out2, case, p0=determine_direction(neighbor,out1,p,p0,v)
                l2[1].append(case)
                l2[2].append(p0)
                face_point.append(l2)
                out=out1
                out1=out2
                ax.scatter(p0[0], p0[1], p0[2], color='black')
        
        elif 0< u < len(point)-1:
            case=find_neighbor(neighbor,out1,out)
            l2=[out1,[case],[p0]]
            p0=point[u]
            p=point[u+1]
            out=face[u]
            l2[2].append(p0)
            out1, case, p0=determine_direction(neighbor,out,p,p0,v)
            l2[1].append(case)
            l2[2].append(p0)
            face_point.append(l2)
            
            while not out1==face[u+1]:
                case=find_neighbor(neighbor,out1,out)
                l2=[out1,[case],[p0]]
                out2, case, p0=determine_direction(neighbor,out1,p,p0,v)
                l2[1].append(case)
                l2[2].append(p0)
                face_point.append(l2)
                out=out1
                out1=out2
                ax.scatter(p0[0], p0[1], p0[2], color='black')
                
                
        elif u==len(point)-1:
            case=find_neighbor(neighbor,out1,out)
            l2=[out1,[case],[p0]]
            p0=point[u]
            p=point[0]
            out=face[u]
            l2[2].append(p0)
            out1, case, p0=determine_direction(neighbor,out,p,p0,v)
            l2[1].append(case)
            l2[2].append(p0)
            face_point.append(l2)
            
            while not out1==face[0]:
                case=find_neighbor(neighbor,out1,out)
                l2=[out1,[case],[p0]]
                out2, case, p0=determine_direction(neighbor,out1,p,p0,v)
                l2[1].append(case)
                l2[2].append(p0)
                face_point.append(l2)
                out=out1
                out1=out2
                ax.scatter(p0[0], p0[1], p0[2], color='black')
            case=find_neighbor(neighbor,out1,out)
            face_point[0][1]=[case]+face_point[0][1]
            face_point[0][2]=[p0]+face_point[0][2]
    
    new_node, del_facet =split_facet(face_point)
    delete_facet(del_facet)
    side_face(new_node_in_circle, new_node)
    
#------------몇번째 네이버인지 확인해 주는 보조함수-------------------------------------------------------------------------+
def find_neighbor(neighbor,i,j):
    if neighbor[i][0]==j:
        case=1
    if neighbor[i][1]==j:
        case=2
    if neighbor[i][2]==j:
        case=3
    return case
    
#------------facet 쪼개는 함수 ----------------------------------------------------------------+
def split_facet(face_point):
    
    many_node=max(node.keys())
    many_facet=max(facet.keys())
    new_node=[]
    del_facet=[]
    i_list=[]
    
    k=1
            
    for i in face_point:
        i_list.append(i[0])
        if len(i[2])==2:
            
            #point = [x for x in [1,2,3] if x not in face_point[i][0]]
            #node[many_node+k]=[2]+list(face_point[i][1][0]) #LEFT CROSS POINT
            node[many_node+k]=[1005]+list(i[2][1]) #RIGHT CROSS POINT
            k+=1

        elif len(i[2])==3:
            #point = [x for x in [1,2,3] if x not in face_point[i][0]]
            #node[many_node+k]=[2]+list(face_point[i][1][0])#LEFT CROSS POINT
            node[many_node+k]=[1005]+list(i[2][1])
            new_node.append(many_node+k)
            node[many_node+k+1]=[1005]+list(i[2][2]) #RIGHT CROSS POINT
            k+=2
            
    last_k=k-1
    for j in range(last_k):
        skin[many_node+1+j]=np.array(node[many_node+1+j][1:])
        
    g=1
    k=1
    prb_i=[]
    for i in face_point:
        if len(i[2])==2:
            del_facet.append(i[0])
            point = [x for x in [1,2,3] if x not in i[1]]
            facet[many_facet+g]=[1005,many_node+k-1,many_node+k,skin_facet[i[0]][point[0]-1]]
            facet[many_facet+g+1]=[1005,many_node+k-1,skin_facet[i[0]][i[1][1]-1],
                                   many_node+k]
            facet[many_facet+g+2]=[1005,skin_facet[i[0]][i[1][1]-1],
                                   skin_facet[i[0]][i[1][0]-1], many_node+k]
            
            k+=1
            g+=3
        
        elif len(i[2])==3:
            if not i[1][0]==i[1][1]: 
                
                if i[0]==face_point[0][0]:
                    del_facet.append(i[0])  
                    point = [x for x in [1,2,3] if x not in i[1]]
                    facet[many_facet+g]=[1005,many_node+last_k,many_node+k,skin_facet[i[0]][point[0]-1]]
                    facet[many_facet+g+1]=[1005,many_node+k,many_node+k+1,skin_facet[i[0]][point[0]-1]]
                    
                    facet[many_facet+g+2]=[1005,many_node+last_k,skin_facet[i[0]][i[1][1]-1],
                                           many_node+k]
                    facet[many_facet+g+3]=[1005,many_node+k,skin_facet[i[0]][i[1][0]-1],
                                           many_node+k+1]
                    facet[many_facet+g+4]=[1005,skin_facet[i[0]][i[1][1]-1],
                                           skin_facet[i[0]][i[1][0]-1],many_node+k]
            
                    k+=2
                    g+=5
                else : 
                    del_facet.append(i[0])
                    point = [x for x in [1,2,3] if x not in i[1]]
                    facet[many_facet+g]=[1005,many_node+k-1,many_node+k,skin_facet[i[0]][point[0]-1]]
                    facet[many_facet+g+1]=[1005,many_node+k,many_node+k+1,skin_facet[i[0]][point[0]-1]]
                    
                    facet[many_facet+g+2]=[1005,many_node+k-1,skin_facet[i[0]][i[1][1]-1],
                                           many_node+k]
                    facet[many_facet+g+3]=[1005,many_node+k,skin_facet[i[0]][i[1][0]-1],
                                           many_node+k+1]
                    facet[many_facet+g+4]=[1005,skin_facet[i[0]][i[1][1]-1],
                                           skin_facet[i[0]][i[1][0]-1],many_node+k]
            
                    k+=2
                    g+=5

                
            elif i[1][0]==i[1][1]: 
                
                if i[0]==face_point[0][0]:
                    del_facet.append(i[0])
                    point = [x for x in [1,2,3] if x not in i[1]]
                    #print(point)
                    facet[many_facet+g]=[1005,skin_facet[i[0]][i[1][0]-1],skin_facet[i[0]][point[0]-1],many_node+k]
                    facet[many_facet+g+1]=[1005,skin_facet[i[0]][i[1][0]-1],skin_facet[i[0]][point[1]-1],many_node+k]
                    facet[many_facet+g+2]=[1005,many_node+last_k,many_node+k,many_node+k+1]
                    
                    length=[]
                    l=[last_k,k+1]
                    for j in range(len(l)):
                        length.append(norm(skin[skin_facet[i[0]][point[0]-1]]-skin[many_node+l[j]]))
                    
                    small=length.index(min(length))
                    big=length.index(max(length))

                    facet[many_facet+g+3]=[1005,many_node+k,skin_facet[i[0]][point[0]-1],
                                                           many_node+l[small]]
                    facet[many_facet+g+4]=[1005,many_node+k,skin_facet[i[0]][point[1]-1],
                                                           many_node+l[big]]
                    
                    prb_i.append([k,g,i[0]])
                    k+=2
                    g+=5
                    
                else :
                    del_facet.append(i[0])
                    point = [x for x in [1,2,3] if x not in i[1]]
                    #print(point)
                    facet[many_facet+g]=[1005,skin_facet[i[0]][i[1][0]-1],skin_facet[i[0]][point[0]-1],many_node+k]
                    facet[many_facet+g+1]=[1005,skin_facet[i[0]][i[1][0]-1],skin_facet[i[0]][point[1]-1],many_node+k]
                    facet[many_facet+g+2]=[1005,many_node+k-1,many_node+k,many_node+k+1]
                    
                    length=[]
                    l=[k-1,k+1]
                    for j in range(len(l)):
                        length.append(norm(skin[skin_facet[i[0]][point[0]-1]]-skin[many_node+l[j]]))
                    
                    small=length.index(min(length))
                    big=length.index(max(length))
                
                    facet[many_facet+g+3]=[1005,many_node+k,skin_facet[i[0]][point[0]-1],
                                                           many_node+l[small]]
                    facet[many_facet+g+4]=[1005,many_node+k,skin_facet[i[0]][point[1]-1],
                                                           many_node+l[big]]
                    
                    prb_i.append([k,g,i[0]])
                    k+=2
                    g+=5
            
    last_g=g-1        
    
    for j in range(last_g):
        skin_facet[many_facet+1+j]=facet[many_facet+1+j][1:]
     
        mass_p=(skin[skin_facet[many_facet+1+j][0]]
        +skin[skin_facet[many_facet+1+j][1]]+skin[skin_facet[many_facet+1+j][2]])/3
        mass_p_dic[many_facet+1+j]=mass_p

    
    for j in range(many_facet+1,many_facet+g):
        revice_facet(j)
    
    #print(prb_i)
    

    for i in prb_i:
        if i[0]==1:
            p=i_list.index(i[2])
            p1=face_point[p-1]
            p2=face_point[p+1]
            ip=p1[0] #=p2[0]
            facet[many_facet+g]=[1005,skin_facet[ip][p1[1][1]-1],many_node+last_k,many_node+2]
            facet[many_facet+g+1]=[1005,skin_facet[ip][p1[1][1]-1],many_node+last_k-1,many_node+last_k]
            facet[many_facet+g+2]=[1005,skin_facet[ip][p1[1][1]-1],many_node+2,many_node+3]
            facet[many_facet+g+3]=[1005,skin_facet[ip][p2[1][1]-1],many_node+last_k-1,many_node+last_k]
            facet[many_facet+g+4]=[1005,skin_facet[ip][p1[1][0]-1],many_node+2,many_node+3]
            
            g+=5
            
            del_facet.append(many_facet+last_g)
            del_facet.append(many_facet+last_g-1)
            del_facet.append(many_facet+last_g-2)
            
            del_facet.append(many_facet+6)
            del_facet.append(many_facet+7)
            del_facet.append(many_facet+8)
            
            
        
        else:
            ki=i[0]
            p=i_list.index(i[2])
            p1=face_point[p-1]
            p2=face_point[p+1]
            ip=p1[0] #=p2[0]
            #p1[1][1]=p2[1][0]
            facet[many_facet+g]=[1005,skin_facet[ip][p1[1][1]-1],many_node+ki-1,many_node+ki+1]
            facet[many_facet+g+1]=[1005,skin_facet[ip][p1[1][1]-1],many_node+ki-2,many_node+ki-1]
            facet[many_facet+g+2]=[1005,skin_facet[ip][p1[1][1]-1],many_node+ki+1,many_node+ki+2]
            facet[many_facet+g+3]=[1005,skin_facet[ip][p2[1][1]-1],many_node+ki-2,many_node+ki-1]
            facet[many_facet+g+4]=[1005,skin_facet[ip][p1[1][0]-1],many_node+ki+1,many_node+ki+2]
            
            g+=5
            
            gi=i[1]
            del_facet.append(many_facet+gi-1)
            del_facet.append(many_facet+gi-2)
            del_facet.append(many_facet+gi-3)
            
            del_facet.append(many_facet+gi+5)
            del_facet.append(many_facet+gi+6)
            del_facet.append(many_facet+gi+7)
        
    return new_node, del_facet

#------------페이셋 순서 수정하는 함수  ----------------------------------------------------------------+
def revice_facet(i) :
    v1=skin[skin_facet[i][1]]-skin[skin_facet[i][0]]
    v2=skin[skin_facet[i][2]]-skin[skin_facet[i][0]]
    n_v=np.cross(v1,v2)
    if not np.dot(n_v,mass_p_dic[i])>0:
        if len(facet)==4:
            facet[i]=[facet[i][0],facet[i][3],facet[i][2],facet[i][1]]
            skin_facet[i]=[skin_facet[i][2],skin_facet[i][1],skin_facet[i][0]]
        elif len(facet)==5:
            facet[i]=[facet[i][0],facet[i][4],facet[i][3],facet[i][2],facet[i][1]]
            skin_facet[i]=[skin_facet[i][3],skin_facet[i][2],skin_facet[i][1],skin_facet[i][0]]
    
#------------옆면 만드는 함수 ----------------------------------------------------------------+
def side_face(new_node_in_circle, new_node):
    many_node=max(node.keys())
    many_facet=max(facet.keys())
    g=0

    for i in range(len(new_node_in_circle)):
        if i<len(new_node_in_circle)-1:
            k=0
            while not new_node[i]+k==new_node[i+1]:            
                facet[many_facet+g+1]=[-1, new_node_in_circle[i], new_node[i]+k, new_node[i]+k+1]
                k+=1
                g+=1
            facet[many_facet+g+1]=[-1, new_node_in_circle[i], new_node[i+1], new_node_in_circle[i+1]]
            g+=1
            
        else : 
            k=0
            while not new_node[i]+k==many_node:
                facet[many_facet+g+1]=[-1, new_node_in_circle[i], new_node[i]+k, new_node[i]+k+1]
                k+=1
                g+=1
                
            facet[many_facet+g+1]=[-1, new_node_in_circle[i], new_node[i]+k, new_node[0]]
            facet[many_facet+g+2]=[-1, new_node_in_circle[i], new_node[0], new_node_in_circle[0]]

#-------------필요없는 부분 제거하는 함수 ----------------------------------------------------------------+
def delete_facet(del_facet):
    del_facet2=set(del_facet)
    
    '''print(len(del_facet))
    print(len(list(del_facet2)))
    if not len(del_facet)==len(list(del_facet2)):
        print('error')'''
    
    for i in list(del_facet2):
        del facet[i]
        del skin_facet[i]
        del mass_p_dic[i]
#-------------관심있는 부분 뽑는 함수 ----------------------------------------------------------------+
def make_interesting_facet(i,detec_l):
    point=skin[i]
    interest_facet={}
    
    for j in mass_p_dic:
        if norm(point-mass_p_dic[j])**2<= detec_l**2:
            interest_facet[j]=skin_facet[j]
    return interest_facet

def make_interesting_skin(i,detec_l):
    point=skin[i]
    interest_skin={}
    for j in skin:
        if norm(point-skin[j])**2<= detec_l**2:
            interest_skin[j]=skin[j]
    return interest_skin
    


print(len(node))
print(len(facet))