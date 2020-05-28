
# -*- coding: utf-8 -*-
"""


@author: Toma
Uni-directional Growth Model Simulator
"""

import random
random.seed(1)
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as sio
import time


#INITIALIZATION
# Define all the parameters
D=1                     # Diffusion coefficientt
delT=1                  #delta T
K1=4*D
K2=(4*math.pi*D)**(3/2)
k0 = 5
k3 = 0.001
k5 = 1

s1 = 0.01               # Sensitivity to concentration gradients from forward direction
s_d1 = 0.05             # Sensitivity to direction perturbation term in forward direction
                    
v0 = 20                 # Base growth rate
v0_grad = 0.008         # Growth rate based on gradient strength
tau = 150.0             # Time constant for growth rate
err1_theta =  math.pi     # Direction perturbation term in theta
v_bi = 1                # Chemical effect of bidirectional growth: 1 at beginning

tau_b = 20              # Time constant for branching rate

P = 0                # Branching probability at infinite: 0 means no branching
B1 = 0.05                # Branching condition 1: Branching may happen if probability value greater than B1
B2 = 0.1               # Branching condition 2: Branching may happen if random uniform term greater than B2

bundle=1                #0 for no bundling, 1 for bundling
chi=0                   #0 for no chirality on the wall, 1 for helix formation
total_step = 70        # Total simulation step: each step is 0.03 days
cell_no = 1000          # Total cell number in forward direction


rad = 100.0             # Inner radius of microcolumn
z_fwd = 0               # Initial seeding position in z direction for forward growth
z_bw = 1200             # Initial seeding position in z direction for backward growth (length of the tube)
  
FibRad=25                #Radius of interaction (RI)


print('CellNo='+str(cell_no))
print('uTennR='+str(rad))
print('uTennZ='+str(z_bw))
print('FibRad='+str(FibRad))
print('bundle='+str(bundle))
print('Chi='+str(chi))
#Create zeros
elevenCol = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
threeCol = np.array([0, 0, 0])
z_growth = threeCol
branch_step = 0

#create 4 cases
if chi==0 and bundle==0:
   case=0

if chi==0 and bundle>0:
   case=1

if chi==1 and bundle==0:
   case=2

if chi==1 and bundle>0:
   case=3

print('Case='+str(case)+' :chi='+str(chi)+', bundle='+str(bundle))

#options={0:c0b0,1:c0b1,2:c1b0,3:c1b1,}


# Define the initial positions
def initial_posXYZ(cell_no, rad, Z):
    ''' it takes a cell number (cell_no, an integer);
        radius for the cylinder (radius, an integer or float);
        Z length of the cylinder (an integer or float)
    and returns a matrix with initial positions.
    
    # Format for the big matrix, aka, column headers
    # 0    1    2     3      4      5       6       7        8              9          10
    # x    y    z   dir_x  dir_y  dir_z  cell_no  step  previous_node  current_node  branch_no
    '''
    pos0XYZ = elevenCol
    for i in range(cell_no):
        rr = 0.999*rad* random.uniform(0.0, 1.0)
        thr= 2*math.pi * random.uniform(0.0, 1.0)
        #zr0= rad * random.uniform(-1.0, 1.0) + Z
        zr0= 0*rad * random.uniform(-1.0, 1.0) + Z
        pos_initial = np.array([[rr*np.cos(thr), rr*np.sin(thr),zr0, 0, 0, 1, i, 0, 0, 0, 1]])

        pos0XYZ = np.vstack((pos0XYZ, pos_initial))
        reset_nodes = elevenCol
        if np.array_equal(reset_nodes, pos0XYZ[0]):
            pos0XYZ = np.delete(pos0XYZ, 0, 0)
    return pos0XYZ



def initial_nodes(cell_no):
    count_nodes = np.array([0])
    for i in range(cell_no):
        initial_node = np.array([0])
        count_nodes = np.vstack((count_nodes, initial_node))
        if i == 0:
            reset_nodes = np.array([0])
            if np.array_equal(reset_nodes, count_nodes[0]):
                count_nodes = np.delete(count_nodes, 0, 0)
    return count_nodes





# Calculate the concentration gradients
def gradientXYZ(pos_previous, pos_current):
    ''' takes a vector of previous positions and
    a vector of current positions and calculates
    the concentration gradients.
    Returns an nx3 array.
    
    K1=4*D; D=diffusion coefficient
    K2=(4*math.pi*D)**(3/2)
    '''
    # Calculate the gradients
    gradXYZ = threeCol
    # gradient components in X, Y and Z
    posx = pos_previous[:, 0] - pos_current[0]
    posy = pos_previous[:, 1] - pos_current[1]
    posz = pos_previous[:, 2] - pos_current[2]
    gr0=(-2/(K1*K2*delT**(5/2)))*posx*np.exp(-(posx**2)/(K1*delT))
    gr1=(-2/(K1*K2*delT**(5/2)))*posy*np.exp(-(posy**2)/(K1*delT))
    gr2=(-2/(K1*K2*delT**(5/2)))*posz*np.exp(-(posz**2)/(K1*delT))
    gradXYZ[0]=np.sum(gr0)
    gradXYZ[1]=np.sum(gr1)
    gradXYZ[2]=np.sum(gr2)
    
    return gradXYZ


def bundle_labels(pos_current,FibRad):
       '''
    Takes current positions (tips of the axons) and the Radius of influence (FIbRad), 
    it detects nearby axons within FibRad and labels them.
    
    Returns dirFiber, a matrix with the number of tips
    at a given time step (rows) and 8 cols (xyz positions components).
       '''
       #how many tips
       current_length = len(pos_current)
       dirFiber=np.zeros((current_length,8))
       #create indices of the tips
       lst_n=np.arange(0,current_length)
       nn=0       
       
       while nn < len(lst_n): 
            nn=0
            #print('Step_No='+str(i))
            n=lst_n[nn]
            
            #calculates the distances from the given tip to all the other tips
            dis1 = np.squeeze(np.sqrt((pos_current[lst_n,0]-pos_current[n,0])**2+(pos_current[lst_n,1]-pos_current[n,1])**2+(pos_current[lst_n,2]-pos_current[n,2])**2))
            
            #find the tips within the radius if interaction of tip n                  
            Rindx0=np.squeeze(np.asarray(np.where((dis1<=FibRad))),axis=0)
            
            #select the indexes of those tips
            Rindx=lst_n[Rindx0]
            
            LRindx=len(Rindx)
            #if there is at least one other tip within FibRad
            if LRindx>1:
               cmx= np.sum(pos_current[Rindx,0])/LRindx+0.1*random.gauss(0,FibRad/16)
               cmy= np.sum(pos_current[Rindx,1])/LRindx+0.1*random.gauss(0,FibRad/16)
               cmz= np.sum(pos_current[Rindx,2])/LRindx
               OB=rad-np.sqrt(cmx**2+cmy**2)
               # check for escapees and correct
               if OB<=0 and cmz<=z_bw:
                  obs=np.sqrt(abs(OB))
                                    
                  cmx=cmx-(3*cmx*obs)/ np.sqrt(cmx**2+cmy**2)
                  cmy=cmy-(3*cmy*obs)/ np.sqrt(cmx**2+cmy**2)
                  
                  #print("New distance= "+str(rad-np.sqrt(cmx**2+cmy**2))) 
               cmz= np.sum(pos_current[Rindx,2])/LRindx
               tvx=cmx-pos_current[Rindx,0]
               tvy=cmy-pos_current[Rindx,1]
               tvz=cmz-pos_current[Rindx,2]
               norm=np.sqrt(tvx**2+tvy**2+tvz**2)
               
               dirFiber[Rindx,0]=tvx/norm
               dirFiber[Rindx,1]=tvy/norm
               dirFiber[Rindx,2]=tvz/norm
               dirFiber[Rindx,3]=cmx
               dirFiber[Rindx,4]=cmy
               dirFiber[Rindx,5]=cmz
               dirFiber[Rindx,6]=rad-np.sqrt(cmx**2+cmy**2)
#                                
               dett=(1./8.)*math.pi*abs(random.uniform(0.0, 1.0))
               dirFiber[Rindx,7]=dett
                             
               #remove Rindx from lst_n:
               lst_n=np.setdiff1d(lst_n,Rindx)
               #print('lst_afterRindx='+str(lst_n))
                    #time.sleep(5)
            
            if n in lst_n:
               lst_n=np.setdiff1d(lst_n,n)
            #print('lst_after_n='+str(lst_n))
       
       return dirFiber 

# Define 4 different ways of handling fibers on the inner surface of the uTenn, depending on chi and bund
def c0b0(pos_current,i,FRG,pos2,l):
    '''Picks the x,y positions for the current tip i;
    pos_current: vector containing xyz positions components.
    FGR is unused.
    pos2: x, y and z position vector (length 3),
    l: not used here;
    
    Case: No chirality, no bundling.
    Returns the updated vector pos2.
    '''
    #print('Chi=0, Bund=0')
    pos2[0] = pos_current[i][0]
    pos2[1] = pos_current[i][1]
    return pos2

def c0b1(pos_current,i,FRG,pos2,l):
    '''Picks the x,y positions for the current tip i;
    pos_current: vector containing xyz positions components.
    FGR: a matrix, dirFiber.
    pos2: x, y and z position vector (length 3).
    l: not used here;
    
    Case: No chirality, bundling.
    Returns the updated vector pos2.
    '''
    #print('Chi=0, Bund=1')
    if np.sum(FRG[i,0:3])==0:            
       pos2[0] = pos_current[i][0]
       pos2[1] = pos_current[i][1]
    else:
       rb=min(FRG[i,6],1.5)
       rx=random.uniform(-rb,rb)
       ry=random.uniform(-np.sqrt(rb**2-rx**2),np.sqrt(rb**2-rx**2))
       
       pos2[0] =FRG[i,3]+rx  
       pos2[1] =FRG[i,4]+ry
    return pos2

def c1b0(pos_current,i,FRG,pos2,l):
    '''Picks the x,y positions for the current tip i;
    pos_current: vector with xyz positions components.
    FGR: a matrix, dirFiber.
    pos2: xy and z position vector (length 3).
    l: z growth;
 
    Case: chirality, No bundling.
    Returns the updated vector pos2.
    '''
    #print('Chi=1, Bund=0')
    t0=np.arctan2(pos_current[i][1],pos_current[i][0]) 
    dett=(1./8.)*math.pi*abs(random.uniform(0.0, 1.0))
   
    pos2[0] = rad*np.cos(t0+dett)
    pos2[1] = rad*np.sin(t0+dett)
    pos2[2] = pos_current[i][2]+l
    return pos2

def c1b1(pos_current,i,FRG,pos2,l):
    '''Picks the x,y positions for the current tip i;
    pos_current: vector with xyz positions components.
    FGR: a matrix, dirFiber.
    pos2: xy and z position vector (length 3).
    l: z growth

    Case: chirality, bundling.
    Returns the updated vector pos2.
    '''
    #print('Chi=1, Bund=1') 
    if np.sum(FRG[i,0:3])==0:            
       t0=np.arctan2(pos2[1],pos2[0]) 
       dett=(1./8.)*math.pi*abs(random.uniform(0.0, 1.0))
       
       pos2[0] = rad*np.cos(t0+dett)
       pos2[1] = rad*np.sin(t0+dett)
       pos2[2] = pos_current[i][2]+l
    else:
       rb=min(FRG[i,6],1.5)
       rx=random.uniform(-rb,rb)
       ry=random.uniform(-np.sqrt(rb**2-rx**2),np.sqrt(rb**2-rx**2))
       t1=np.arctan2(FRG[i,4],FRG[i,3])
       dett=FRG[i,7]
       pos2[0] = rad*np.cos(t1+dett)+rx
       pos2[1] = rad*np.sin(t1+dett)+ry
       pos2[2] = pos_current[i][2]+l
    return pos2
                  

#Find the new position in forward direction
def generate_nextXYZ(step_no, pos_current, pos_previous, v_bi, bundle):
    '''
    step_no: current step number;
    pos_current: current positions vector (11-columns);
    pos_previous: previous positions vector (11-columns);
    v_bi: growth rate;
    bundle: a flag, 0 No bundling, 1 for bundling.
    
    Returns an updated positions vector (11 columns).
    '''
    pos_new_full = elevenCol
    i = 0
    
    current_length = len(pos_current)
    
    if bundle>0:
       FRG=bundle_labels(pos_current,FibRad)
    else:
       FRG=0
        
    while i < current_length:
        
        step = v0 * random.uniform(0.8, 1.2)
        #get dir1
        dir1 = np.array(pos_current[i][3:6])
        #estimate concentration gradients
        grad = gradientXYZ(pos_previous[:, 0:3], pos_current[i][0:3])
        
        #err1 = np.array([random.uniform(-1.0, 1.0), random.gauss(0, err1_theta), random.gauss(0.0, 1)])
        #err1= np.array([random.uniform(-0.5, 0.5), math.pi*random.uniform(-1,1), random.gauss(0.0, 1)])
        #create noise perturbations
        errR=random.uniform(0.0, 0.1)
        errTh=2*math.pi*random.uniform(0.0,1.0)
        errZ=random.uniform(0.0,0.01)
        err1= np.array([errR*np.cos(errTh), errR*np.sin(errTh), errZ])
        #if there is bundling/attraction
        if bundle>0:
           dirF=np.array([FRG[i,0],FRG[i,1],FRG[i,2]]) #attraction
        else:
           dirF=threeCol # no attraction
        if pos_current[i][2]>=z_bw:
           dirF=threeCol #no attraction outside of the cylinder
           
        #Normalizing the gradient
        strength = math.sqrt(grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2)
        if strength > 0:
            grad1 = grad / strength
        else:
            grad1 = threeCol
            grad = threeCol

        #allows different growth modes in and out of the cylinder. Here we use the same,
        # however s1 and s_d1 can have different values inside and outside 
        if pos_current[i][2]>=z_bw:
               dir2 = dir1 + s1 * grad1 + s_d1 * err1
        else:
               dir2 = dir1 + s1* grad1 + s_d1 * err1
        
        l1 = math.sqrt(dir2[0] ** 2 + dir2[1] ** 2 + dir2[2] ** 2)
        dir2 = dir2 / l1
        #Z growth  for a given tip
        l = (v_bi * (v0_grad * strength + step)) * 2 ** (-step_no / tau)
        #updating the new xyz positions of the tip
        pos2 = pos_current[i][0:3] +dir2* l+ dirF*(5*s_d1)*bundle*l
        ########################################################
        #check for overlap and repositioning if it exists
        ix=(pos2[0] - pos_current[:][0] == 0)
        iy=(pos2[1] - pos_current[:][1] == 0)
        iz=(pos2[2] - pos_current[:][2] == 0)
        if np.sum(ix*iy*iz> 0):
                    print('Overlap Warning! Re-calculating x, y and z positions')
                    errR = random.uniform(0.0, 0.1)
                    errTh = 2*math.pi*random.uniform(0.0,1.0)
                    errZ = random.uniform(0.0,0.01)
                    err1 = np.array([errR*np.cos(errTh), errR*np.sin(errTh), errZ])
                    dir2 = dir1 + s1 * grad1 + s_d1 * err1
                    l1 = math.sqrt(dir2[0] ** 2 + dir2[1] ** 2 + dir2[2] ** 2)
                    dir2 = dir2 / l1
                    l = (v_bi * (v0_grad * strength + step)) * 2 ** (-step_no / tau)
                    pos2 = pos_current[i][0:3] +dir2* l+ dirF*(5*s_d1)*bundle*l
                    
        
        ##############################################
        OBT=rad-np.sqrt(pos2[0]**2+pos2[1]**2)
        # check for escapee tips and correct
        if OBT<=0 and pos2[2]<=z_bw:
           obsT=np.sqrt(abs(OBT))
           Tx=pos2[0]
           Ty=pos2[1]
           disT=np.sqrt(Tx**2+Ty**2)
           Tx=Tx-(3*Tx*obsT)/(disT)
           Ty=Ty-(3*Ty*obsT)/(disT)
           pos2[0]=Tx
           pos2[1]=Ty 
           if rad-np.sqrt(Tx**2+Ty**2)<0:
              print("Escapee tip. New distance= "+str(rad-np.sqrt(Tx**2+Ty**2))) 
        ##############################################
                
        cell_number = pos_current[i][6]
        node_counter = initial_node[int(cell_number)]

        # make uTENN grow out after reaching the end
        if pos2[2] <= z_bw:
            #if abs(pos2[0]) <= rad:
            if math.sqrt(pos2[0]**2+pos2[1]**2) <= rad:
                
                pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter + 1, pos_current[i][10]))
                initial_node[int(cell_number)] = node_counter + 1
                pos_new_full = np.vstack((pos_new_full, pos2_full))
                reset_nodes = elevenCol
                if np.array_equal(reset_nodes, pos_new_full[0]):
                    pos_new_full = np.delete(pos_new_full, 0, 0)
            else:
                pos2=options[case](pos_current,i,FRG,pos2,l)

                pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter + 1, pos_current[i][10]))
                initial_node[int(cell_number)] = node_counter + 1
                pos_new_full = np.vstack((pos_new_full, pos2_full))
                reset_nodes = elevenCol
                if np.array_equal(reset_nodes, pos_new_full[0]):
                    pos_new_full = np.delete(pos_new_full, 0, 0)
        else:
            pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter + 1, pos_current[i][10]))
            initial_node[int(cell_number)] = node_counter + 1
            pos_new_full = np.vstack((pos_new_full, pos2_full))
            reset_nodes = elevenCol
            if np.array_equal(reset_nodes, pos_new_full[0]):
                pos_new_full = np.delete(pos_new_full, 0, 0)
        
        i += 1

    return pos_new_full

def plotter(pos_all, rad, z_bw):
    '''
    Takes all the positions, the radius and 
    height (z_bw) and draws a figure with the cells
    and cylinder.
    '''
    fig1 = plt.figure(figsize=(30, 30))
    ax3d = fig1.add_subplot(111, projection='3d')
    
    x = pos_all[:, 0] 
    y = pos_all[:, 1] 
    z = pos_all[:, 2]
    ax3d.plot(x, y, z, 'r.',markersize=3)
    
    # Cylinder
    x_center=0
    y_center=0
    z_0c=0
    
    height=z_bw
    resolution=100
    color='b'
    xc = np.linspace(x_center-rad, x_center+rad, resolution)
    zc = np.linspace(z_0c, z_0c+height, resolution)
    X, Z = np.meshgrid(xc, zc)
    Y = np.sqrt(rad**2 - (X - x_center)**2) + y_center 

    ax3d.plot_surface(X, Y, Z, linewidth=0, alpha=0.2, color=color)
    ax3d.plot_surface(X, (2*y_center-Y), Z, linewidth=0, alpha=0.2,color=color)
    
    #seeds
    x_init = pos0[:, 0] 
    y_init = pos0[:, 1] 
    z_init = pos0[:, 2]
    ax3d.plot(x_init, y_init, z_init, 'b.',markersize=3)
    
    #ax3d.set_aspect(1)
    ax3d.set_xlim([-rad, rad])
    ax3d.set_ylim([-rad, rad])
    ax3d.set_zlim([-10, z_bw+10])
    ax3d.set_xlabel(r'x, $ \mu m$', fontsize=15, labelpad=20)
    ax3d.set_ylabel(r'y, $ \mu m$', fontsize=15, labelpad=20)
    ax3d.set_zlabel(r'z, $ \mu m$', fontsize=15, labelpad=30) 
    ax3d.set_zticks(np.arange(0, z_bw, 100))
    ax3d.set_xticks(np.arange(-rad, rad, 50))
    ax3d.set_yticks(np.arange(-rad, rad, 50))
    ax3d.set_title('%.2f Days' % (step_no * 0.03), fontsize=30)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='vertical')
    ax3d.view_init(elev=4, azim=315)
    #plt.savefig('Bundle_50Fan' , dpi=500, bbox_inches='tight', pad_inches=0.1)
    #plt.savefig('Step_%d.png' % (step_no + 1), dpi=400, bbox_inches='tight', pad_inches=1)
    plt.show()


def pos_saver(pos_all):
    ''' Takes the positions matrix and saves it in
    .mat (MATLAB) and .txt formats.
    '''
    name_file_fwd = 'UGM'+'_B_'+str(bundle)+'_Ch_'+str(chi)+'_Br_'+str(P)+'_R_'+str(rad)+'_Z_'+str(z_bw) + '_' + str(cell_no)
    name_file_fwd_txt = name_file_fwd + '.txt'
    name_file_fwd_mat = name_file_fwd + '.mat'
    
    np.savetxt(name_file_fwd_txt, pos_all)
    sio.savemat(name_file_fwd_mat, {'pos_all': pos_all})
    




options={0:c0b0,1:c0b1,2:c1b0,3:c1b1,}

#timing/Start
#starttime = time.clock()
starttime = time.perf_counter()

pos0 = initial_posXYZ(cell_no, rad, z_fwd)
initial_node = initial_nodes(cell_no)

# From step 0 to step 1, pos_current = pos_previous
step_no = 1
pos_1 = generate_nextXYZ(step_no, pos0, pos0, v_bi, bundle)

# After step 1, pos_current different from pos_previous
pos_all = np.vstack((pos0, pos_1))
pos_current = pos_1
pos_previous = pos0
#go through the rest of the steps
for step_no in range(2, total_step):
    print('Step_No='+str(step_no))
    original_length = len(pos_current)
    a = 0
    while a < len(pos_current):
        if pos_current[a][2]<=z_bw:
           p=0
        else: 
           p = P*(1 - math.exp(-( pos_current[a][2]-z_bw) / (tau_b))) 
           #p = P * (1 - math.exp(-(step_no - branch_step) * 0.3 / (tau_b)))
        if (p >= B1 and random.uniform(0, 1) > B2):
            insert_array = np.hstack((pos_current[a][0:10], 2))
            pos_current = np.insert(pos_current, a + 1, insert_array, 0)
            a += 1
        a += 1
    if len(pos_current) > original_length:
        branch_step = step_no
    else:
        branch_step = branch_step

    pos_new = generate_nextXYZ(step_no, pos_current, pos_previous, v_bi, bundle)
    pos_all = np.vstack((pos_all, pos_new))

    for m in range(len(pos_new)):
        pos_new[m] = np.hstack((pos_new[m][0:10], 1))

    pos_previous = pos_current
    pos_current = pos_new

    
    ########################################################################

#     GrowthRateMean = GrowthRate(pos_current,pos_current_b)
#     z_growth = np.vstack((z_growth, GrowthRateMean))
#     reset_nodes = np.array([0, 0, 0])
#     if np.array_equal(reset_nodes, z_growth[0]):
#         z_growth = np.delete(z_growth, 0, 0)  # save the longest tips of forward and backward
#
# GrowthRateSaver(z_growth) #check!

    ########################################################################

# Print running time
#endtime = time.clock()
endtime = time.perf_counter()
print('Loop time = ' + str(endtime - starttime) + 's')

pos_saver(pos_all)
plotter(pos_all,rad,z_bw)


# Plot cross sections along z
#st_ind=pos_all[:,7]
#max_step_no=int(max(st_ind))
#print(max_step_no)
#step_no=10
#ti=np.where(st_ind==step_no)
#ti1=ti[0] 
#x1 = pos_all[ti1, 0] 
#y1 = pos_all[ti1, 1] 
#z1 = pos_all[ti1, 2]
#fig1 = plt.figure(figsize=(25, 25))
#plt.plot(x1, y1, 'b.',markersize=30)
#plt.axis('equal')
#plt.title('Cross section from simulation')

######################  VTK ##############################
# This file is used to write vtk format for  Growth Model
# Remember to change the input text files name

# import math
# import datetime

# print(datetime.datetime.now())

# cellnumber = []         # Cell number: saved in column 6
# stepnumber = []         # Step number: saved in column 7
# previousnode = []       # Previous node number: saved in column 8
# currentnode = []        # Current node number: saved in column 9

# uniquenumber = []
# uniquenumber1 = []

# x_fwd = []              # x coordinates
# y_fwd = []              # y coordinates
# z_fwd = []              # z coordinates

# new_row = []

# # Open the output from Growth model and save in a list; Remember to change the file's name
# print('b')
# with open('UGM_B_1_Ch_0_Br_0_R_100.0_Z_1200_500.txt') as f:

#     content = f.readlines()

# # Read all useful information from the list
# for line in content:
#     currentnode.append(line.split()[9])
#     previousnode.append(line.split()[8])
#     cellnumber.append(line.split()[6])
#     uniquenumber.append(line.split()[6] + '-' + line.split()[9])       # Create unique numbers for cell number and current node
#     uniquenumber1.append(line.split()[6] + '-' + line.split()[8])      # Create unique numbers for cell number and previous node
#     stepnumber.append(line.split()[7])
    
#     x_fwd.append(float(line.split()[0]))
#     y_fwd.append(float(line.split()[1]))
#     z_fwd.append(float(line.split()[2]))
    

# print('c')
# x_fwd = [i for i in x_fwd]
# y_fwd = [j for j in y_fwd]
# z_fwd = [k for k in z_fwd]

# # print('d, uniquenumber1 lenght: ', len(uniquenumber1))

# # from multiprocessing import Pool

# # def indexing(n):
# #     global uniquenumber
# #     return uniquenumber.index(n)

# # start = time.time()
# # p = Pool()
# # new_row = p.map(indexing, uniquenumber1)
# # p.close()
# # p.join()
# # end = time.time()
# # print('Done!')
# # print('Time: ', end-start)

# ####################### Start to write VTK format ############################
# stepnumber = [int(float(i)) for i in stepnumber]
# cellnumber = [int(float(i)) for i in cellnumber]


# NoStep = max(stepnumber)
# NoCell = max(cellnumber)
# CellType = 3                    # In VTK format, Type 3 is VTK_LINE
# #CellType = 1                    # In VTK format, Type 1 is VTK_pt

# counter = 0

# while counter < NoStep:
  
#     print(counter)
#     new_x = []
#     new_xb = []
#     new_yb = []
#     new_y = []
#     new_z = []
#     new_zb = []
#     new_connection = []
#     new_connectionb = []

#     for i,s in enumerate(stepnumber):

#         if s <= counter:
#             new_x.append(x_fwd[i])
#             new_y.append(y_fwd[i])
#             new_z.append(z_fwd[i])
#             #new_connection.append(new_row[i])
#             new_connection.append(new_row[i])
            
#         else:
#             totalx = new_x + new_xb
#             totaly = new_y + new_yb
#             totalz = new_z + new_zb
#             totalconnection = new_connection + [r + len(new_connection) for r in new_connectionb]
#             with open('UGM%s.vtk' %s,'wb') as output:
#                 output.write("# vtk DataFile Version 3.0\n".encode('ascii'))
#                 output.write(("%i Cells\n" %(NoCell+1)).encode('ascii'))
#                 output.write("ASCII\n".encode('ascii'))
#                 output.write("DATASET UNSTRUCTURED_GRID\n".encode('ascii'))
#                 output.write(("POINTS %d FLOAT\n" % (len(totalx))).encode('ascii'))
#                 for i in range(len(totalx)):
#                     output.write(("%s %s %s\n" % (str(totalx[i]), str(totaly[i]),str(totalz[i]))).encode('ascii'))
#                 output.write(("CELLS %d %d\n".encode('ascii') % (len(totalconnection), len(totalconnection) * 3)))
#                 for j in range(len(totalconnection)):
#                     output.write(("%d %s %s\n" % (2, str(j), str(totalconnection[j]))).encode('ascii'))
#                 output.write("CELL_TYPES %d\n".encode('ascii') % (len(totalconnection)))
#                 for m in range(len(totalconnection)):
#                     output.write(("%d\n" % CellType).encode('ascii'))
#                 output.write("CELL_DATA %d\n".encode('ascii') % (len(totalx)))
#                 output.write("SCALARS cell_scalars int 1\nLOOKUP_TABLE default\n".encode('ascii'))
#                 for n in range(len(new_x)):
#                     output.write("1\n".encode('ascii'))
#                 for p in range(len(new_xb)):
#                     output.write("1\n".encode('ascii'))

#             counter +=1
#             break

# print(datetime.datetime.now())