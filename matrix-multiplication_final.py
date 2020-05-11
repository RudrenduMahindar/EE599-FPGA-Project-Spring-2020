#!/usr/bin/env python
# coding: utf-8

# # Matrix Multiplication
# 
# ADD YOUR COMMENT

# In[157]:


import time
import numpy as np
from pynq import Xlnk, Overlay, allocate


# In[158]:


overlay = Overlay("mmult.xclbin")


# In[159]:


#taken a fixed number fixed_num in input matrices to avoid overflow
size = 4096
fixed_num = 4 
in_buffer1 = allocate((size, size), np.uint32)
in_buffer2 = allocate((size, size), np.uint32)
out_buffer = allocate((size, size), np.uint32)


# In[160]:


# check the name of filename.xclbin, use overlay.filename_1 to assign the kernel IP to a variable called multiplier
multiplier = overlay.mmult_1
multiplier.signature


# In[161]:


for i in range(size):
    for j in range(size):
        in_buffer1[i][j]=fixed_num
        in_buffer2[i][j]=fixed_num
        out_buffer[i][j]=0
        
# Before we can start the kernel we need to make sure that the buffers are synced to the FPGA card


# In[162]:


get_ipython().run_cell_magic('timeit', '', 'in_buffer1.sync_to_device()\nin_buffer2.sync_to_device()\n\nmultiplier.call(in_buffer1,in_buffer2,out_buffer,size,size,size)\n\nout_buffer.sync_from_device()')


# In[163]:


get_ipython().run_cell_magic('timeit', ' #communication time for input buffers', 'in_buffer1.sync_to_device()\nin_buffer2.sync_to_device()')


# In[164]:


get_ipython().run_cell_magic('timeit', '#compute time', 'multiplier.call(in_buffer1,in_buffer2,out_buffer,size,size,size)')


# In[165]:


get_ipython().run_cell_magic('timeit', ' #communication time for output buffer', 'out_buffer.sync_from_device()')


# In[166]:


start_time = time.time()
in_buffer1.sync_to_device()
in_buffer2.sync_to_device()

multiplier.call(in_buffer1,in_buffer2,out_buffer,size,size,size)

out_buffer.sync_from_device()
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))


# In[167]:


out_buffer[50][50]


# In[126]:


# deallocate the buffers and free the FPGA context using Overlay.free
get_ipython().run_line_magic('xdel', 'in_buffer1')
get_ipython().run_line_magic('xdel', 'in_buffer2')
get_ipython().run_line_magic('xdel', 'out_buffer')
overlay.free()


# In[148]:


A=np.zeros((size,size),np.uint32)
B=np.zeros((size,size),np.uint32)
C=np.zeros((size,size),np.uint32)


# In[149]:


for i in range(size):
    for j in range(size):
        A[i][j]=fixed_num
        B[i][j]=fixed_num
        C[i][j]=0


# In[129]:


def mat_mult(A,B,C):
    for i in range(size): 
        for j in range(size): 
            for k in range(size): 
                C[i][j] += A[i][k] * B[k][j] 


# In[95]:


start_time = time.time()
mat_mult(A,B,C)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))


# In[96]:


C[50][50]


# In[130]:


def block_mat_mult(A,B,C,blocksize):
    for block_i in range(0,size,blocksize):
        for block_j in range(0,size,blocksize):
            for block_k in range(0,size,blocksize): 
                for i in range(blocksize): 
                    for j in range(blocksize): 
                        for k in range(blocksize): 
                            C[block_i+i][block_j+j] += A[block_i+i][block_k+k] * B[block_k+k][block_j+j]
                        
                        
                        


# In[98]:


#Following section starts block matrix multiply with different block sizes


blocksize=2
C=np.zeros((size,size),np.uint32)

start_time = time.time()
block_mat_mult(A,B,C,blocksize)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))


# In[99]:


C[50][50]


# In[100]:


blocksize=4
C=np.zeros((size,size),np.uint32)

start_time = time.time()
block_mat_mult(A,B,C,blocksize)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))


# In[101]:


C[50][50]


# In[102]:


blocksize=8
C=np.zeros((size,size),np.uint32)

start_time = time.time()
block_mat_mult(A,B,C,blocksize)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))


# In[103]:


C[50][50]


# In[104]:


blocksize=16
C=np.zeros((size,size),np.uint32)

start_time = time.time()
block_mat_mult(A,B,C,blocksize)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))


# In[105]:


C[50][50]


# In[131]:


blocksize=32
C=np.zeros((size,size),np.uint32)

start_time = time.time()
block_mat_mult(A,B,C,blocksize)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))


# In[132]:


C[50][50]


# In[133]:


blocksize=64
C=np.zeros((size,size),np.uint32)

start_time = time.time()
block_mat_mult(A,B,C,blocksize)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))


# In[134]:


C[50][50]


# In[135]:


blocksize=128
C=np.zeros((size,size),np.uint32)

start_time = time.time()
block_mat_mult(A,B,C,blocksize)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))


# In[112]:


blocksize=256
C=np.zeros((size,size),np.uint32)

start_time = time.time()
block_mat_mult(A,B,C,blocksize)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))


# In[150]:


#numpy function

C=np.zeros((size,size),np.uint32)

start_time = time.time()
np.matmul(A,B,C)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))


# In[151]:


C[50][50]


# In[ ]:




