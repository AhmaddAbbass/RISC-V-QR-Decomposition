.data
zero_float: .float 0.0
matrixA:    .float 1.0,2.0,3.0,4.0,1.0,2.0,3.0,4.0,1.0,2.0,3.0,4.0,1.0,2.0,3.0,4.0  # 32x32 floats (4*4*4 bytes)
matrixQ:    .float 1.0,2.0,3.0,4.0,1.0,2.0,3.0,4.0,1.0,2.0,3.0,4.0,1.0,2.0,3.0,4.0   # Space for the resulting orthogonal matrix Q
matrixR:    .float 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0  # Space for the resulting upper triangular matrix R
q_k:        .float 0.0,0.0,0.0,0.0
q_j:        .float 0.0,0.0,0.0,0.0

.text

main:
    la x10, matrixA      # Load address of matrixA into x10
    la x11, matrixQ      # Load address of matrixQ into x11
    la x14,matrixR
    la x17,q_k
    la x22,q_j
    li x12, 4      # Number of rows (32)
    li x13, 4 
    ##li x19,2         # Number of columns (32)
         # Loop back if not end of matrix
    call modified_granny    # Call the Gram-Schmidt function

    # Assuming you need to print the matrices or verify them, add your specific code here
    # This could include printing the elements of matrixQ and matrixR to a terminal or file
copy_matrices: ## A in x10, Q in x11
    addi sp, sp, -32
    sw x6, 0(sp)
    sw x5, 8(sp)
    sw x1, 16(sp)
    sw x20,24(sp)
      # i = 0
    ## suppose loop limit is in x19, loading in main  # Loop limit for i and j, since matrix is 32x32
    
    ## knowing nbr of rows is in x12 , we shift left
    ## by logbase2 of number of rows hence
    ## call logbase2 function with x12 input 
    ## logbase2 takes input in x14 and produce output in x11
    ## hence need save x11 register from danger
    addi x20,x11,0
    addi x14,x12,0
    call logbase2
    ## now x11 has the value to shift , put it in x20
    addi x5,x11,0
    addi x11,x20,0
    addi x20,x5,0 
    li x5, 0

loopd:
    beq x5,x19,exit_loopd
    li x6, 0  # j = 0
loope:
    beq x6,x19,exit_loope
    sll x28, x5, x20  # x28 = i * 32 (offset for ith row)
    add x7, x28, x6  # x7 = i*32 + j (offset for [i][j])
    slli x7, x7, 2   # x7 = (i*32 + j) * 4 (scale for single precision)
    add x9, x10, x7  # x9 = &A[i][j]
    flw f8, 0(x9)    # Load A[i][j]
    add x9, x11, x7  # x9 = &Q[i][j]
    fsw f8, 0(x9)    # Store into Q[i][j] = A[i][j]

    addi x6, x6, 1   # j++
    j loope
exit_loope:
    addi x5, x5, 1   # i++
    j loopd
exit_loopd:
    lw x5, 8(sp)
    lw x6, 0(sp)
    lw x1, 16(sp)
    lw x20,24(sp)
    addi sp, sp, 32
    li a0,10
    ecall



#----------------------------------------NORM--------------------------------------------------

# Calculates the norm of a vector in single precision
# Registers:
# x10: Base address of the vector
# x11: Size of the vector
# Result: f0 (Norm of the vector in single precision)

norm:
    addi sp, sp, -48
    fsw f0, 0(sp)  # Save f0 on the stack
    fsw f2, 8(sp)  # Save f2 on the stack
    sw x12, 16(sp)  # Save x12 on the stack
    sw x13, 24(sp)  # Save x13 on the stack
    sw x14, 32(sp)  # Save x14 on the stack
    sw x1, 40(sp)  # Save x1 on the stack
    la x13, zero_float  # Load address of zero_float into x13 (defined elsewhere in memory)
    flw f2, 0(x13)  # Load 0.0f into f2
    li x12, 0  # Initialize loop counter

norm_loop:
    bge x12, x11, compute_sqrt  # If loop counter >= vector size, exit loop

    slli x13, x12, 2  # Calculate offset for single precision (4 bytes per element)
    add x14, x10, x13  # Calculate element address
    flw f0, 0(x14)  # Load vector element into f0
    fmul.s f0, f0, f0  # Square the element
    fadd.s f2, f2, f0  # Accumulate in f2

    addi x12, x12, 1  # Increment loop counter
    j norm_loop  # Jump back to start of loop

compute_sqrt:
    fsqrt.s x0, f2  # Take square root of sum in f2, store result in f0
    flw f2, 8(sp)  # Restore f2
    lw x12, 16(sp)  # Restore x12
    lw x13, 24(sp)  # Restore x13
    lw x14, 32(sp)  # Restore x14
    lw x1, 40(sp)  # Restore x1
    addi sp, sp, 48  # Adjust stack pointer back
    ret  # Return from function





#-------------------------------------------------------DOT--------------------------------------------

# Calculates the dot product of two vectors in single precision
# Registers:
# x10: Base address of vector1
# x11: Base address of vector2
# x12: Size of the vectors
# Result: f0 (Dot product in single precision)

dot:
    addi sp, sp, -56
    fsw f0, 0(sp)
    fsw f1, 8(sp)
    fsw f2, 16(sp)
    sw x13, 24(sp)
    sw x14, 32(sp)
    sw x15, 40(sp)
    sw x16, 48(sp)
    sw x1, 56(sp)
    la x28, zero_float
    flw f2, 0(x28)
    li x13, 0

dot_loop:
    bge x13, x12, end_dot

    slli x14, x13, 2
    add x15, x10, x14
    add x16, x11, x14

    flw f0, 0(x15)
    flw f1, 0(x16)
    fmul.s f0, f0, f1
    fadd.s f2, f2, f0

    addi x13, x13, 1
    j dot_loop

end_dot:
    fmv.s f0, f2
    flw f1, 8(sp)
    flw f2, 16(sp)
    lw x13, 24(sp)
    lw x14, 32(sp)
    lw x15, 40(sp)
    lw x16, 48(sp)
    lw x1, 56(sp)
    addi sp, sp, 56
    ret



#-------------------------------------------------------------------COPY MATRICES-------------------------------

# Copies one matrix to another in single precision
# Registers:
# x10: Base address of source matrix A
# x11: Base address of destination matrix Q
# Assumes matrices are 32x32


#---------------------------------------------------------------CLASSICAL GRAM-SCHMIDT------------------------------------


# Gram-Schmidt orthogonalization in single precision
# Registers:
# x10: Base address of matrix A
# x11: Base address of matrix Q
# x12: Number of rows (M)
# x13: Number of columns (N)
## 4x4

## save value to shift left with in x31 
## logbase2 take input inx14 and produce output in x11
granny:
    addi x16,x11,0
    addi x30,x14,0 ## used x16 and x30 as temps because later will be used as counter
    addi x14,x12,0 ## nbr of rows in x12 need row size
    addi x15,x12,0
    call logbase2 
    addi x31,x11,0 ## now final value in x31 
    addi x11,x16,0 ## old val x11 is back
    addi x14,x30,0 ## old val x14 is back
    addi x12,x15,0
    li x15, 0  # k = 0
loop1:
    bge x15, x12, end_loop1
    li x16, 0  # i = 0

loop2:
    bge x16, x13, end_loop2
    slli x19, x16, 2
    add x19, x17, x19  # &q[i]
    sll x20, x16, x31
    add x20, x20, x15
    slli x20, x20, 2
    add x20, x20, x10  # &A[i][k]
    flw f1, 0(x20)
    fsw f1, 0(x19)  # q[i] = A[i][k]
    addi x16, x16, 1
    j loop2
end_loop2:
    li x30, 0  # j = 0
loop3:
    bge x30, x15, end_loop3
    li x16, 0
loop4:
    bge x16, x13, end_loop4
    slli x19, x16, 2
    add x19, x22, x19  # &q_j[i]
    sll x20, x16, x31
    add x20, x20, x30
    slli x20, x20, 2
    add x20, x20, x11  # &Q[i][j]
    flw f1, 0(x20)
    fsw f1, 0(x19)  # q_j[i] = Q[i][j]
    addi x16, x16, 1
    j loop4
end_loop4:
    addi x5, x10,0
    addi x6,x31,0
    addi x31,x11,0
    addi x10, x22, 0
    addi x11, x17, 0
    jal dot  # Call dot product between q_j and q_k
    addi x10, x5, 0
    addi x11, x31, 0
    addi x31,x6,0

    sll x20, x30, x31
    add x20, x15, x20
    slli x20, x20, 2
    add x20, x20, x14  # &R[j][k]
    fsw f0, 0(x20)  # R[j][k] = result of dot product
    addi x16, x0, 0
loop5:
    bge x16, x13, end_loop5
    slli x19, x16, 2
    add x19, x17, x19  # &q_k[i]
    sll x20, x16, x31
    add x20, x20, x30
    slli x20, x20, 2
    add x20, x20, x11  # &Q[i][j]
    flw f1, 0(x20)  # f1 = Q[i][j]
    fmul.s f1, f1, f0  # f1 = f1 * f0
    flw f2, 0(x19)
    fsub.s f2, f2, f1
    fsw f2, 0(x19)
    addi x16, x16, 1
    j loop5
end_loop5:
    addi x30,x30, 1
    j loop3
end_loop3:
    addi x5, x10, 0
    addi x6,x31,0
    addi x31, x11, 0
    addi x10, x17, 0
    addi x11, x12, 0
    jal norm  # Call norm of q_k
    addi x10,x5,0
    addi x11,x31,0
    addi x31,x6,0

    sll x20, x15, x31
    add x20, x20, x15
    slli x20, x20, 2
    add x20, x20, x14  # &R[k][k]
    fsw f0, 0(x20)  # R[k][k] = norm(q_k)

        # Assume x5 is available for use
    la x7, zero_float  # Load address of zero_float (float 0.0) into x5
    flw f1, 0(x7)      # Load 0.0f into f1 for comparison
    feq.s x7, f0, f1   # Set x5 to 1 if f0 is equal to 0.0, otherwise set to 0

    bnez x7, end_branch # Branch if x5 is not zero, i.e., f0 is not zero

    li x16, 0
loop6:
    bge x16, x13, end_loop6
    sll x20, x16, x31
    add x20, x20, x15
    slli x20, x20, 2
    add x20, x20, x11  # &Q[i][k]
    slli x19, x16, 2
    add x19, x17, x19  # &q_k[i]
    flw f2, 0(x19)
    fdiv.s f2, f2, f0
    fsw f2, 0(x20)  # Q[i][k] = q_k[i] / R[k][k]
    addi x16, x16, 1
    j loop6
end_loop6:
end_branch:
    addi x15, x15, 1 #k++
    j loop1
end_loop1:
    li  a0,10
    ecall

## 2x2 granny
modified_granny:
# Modified Gram-Schmidt orthogonalization in single precision
# Registers:
# x10: Base address of matrix A
# x11: Base address of matrix Q
# x12: Number of rows (M)
# x13: Number of columns (N)
# x17: q_k's
# x22: q_j's
## x15: k
# x16 :i
# x30: j
addi x16,x11,0
addi x30,x14,0 ## used x16 and x30 as temps because later will be used as counter
addi x14,x12,0 ## nbr of rows in x12 need row size
addi x15,x12,0
call logbase2 
addi x31,x11,0 ## now final value in x31 
addi x11,x16,0 ## old val x11 is back
addi x14,x30,0 ## old val x14 is back
addi x12,x15,0
addi x15,x0,0
lop1:
bge x15,x13,end_lop1 
addi x16,x0,0
lop2:
bge x16,x12,end_lop2
slli x19,x16,2 ## i*4
add x19,x17,x19  ## & q_k[i]
sll x20,x16,x31 ## i*4
add x20,x20,x15 ## i*4+k
slli x20,x20,2 ## [i][k]
add x20,x20,x11 ## &Q[i][k]
flw f0,0(x20)
fsw f0,0(x19)
addi x16,x16,1
j lop2
end_lop2:

## need to find norm(q_k) ,q_k in x17
addi x5,x10,0 ## x5=x10
addi x6,x31,0
addi x31,x11,0 ## x31=x11
addi x10,x17,0
addi x11,x12,0
jal norm
addi x10,x5,0
addi x11,x31,0
addi x31,x6,0
## now f0 holds the norm of q_k
la x7, zero_float  # Load address of zero_float (float 0.0) into x5
flw f1, 0(x7)      # Load 0.0f into f1 for comparison
feq.s x7, f0, f1   # Set x5 to 1 if f0 is equal to 0.0, otherwise set to 0

bnez x7, end_lop3 # Branch if x5 is not zero, i.e., f0 is not zero


addi x16,x0,0
lop3:
bge x16,x13,end_lop3
sll x19,x16,x31 ## i*4
add x19,x19,x15 ## i*4+k
slli x19,x19,2 ## [i][k]
add x20,x19,x11 ## x20=&Q[i][k]
flw f1,0(x20)  ## f1=Q[i][k]
fdiv.s f1,f1,f0 ## Q[i][k]/R[k][k]
fsw f1,0(x20) ## updated Q[i][k]
addi x16,x16,1
j lop3
end_lop3:
addi x30,x15,1 ## j=k+1
lop4:
bge x30,x13,end_lop4
sll x20,x15,x31 ## k*4
add x20,x20,x30 ## k*4+j
slli x20,x20,2 ## [k][j]
add x20,x20,x14 ## &R[k][j]
la x7,zero_float
flw f5,0(x7)
fsw f5,0(x20) ## R[k][j]=0
addi x16,x0,0 ## i=0
lop5: ## x20=&R[k][j]
bge x16,x13,end_lop5
sll x19,x16,x31 ## x19=i*4
add x22,x19,x15 ## x20=i*4+k
slli x22,x22,2 ## [i][k]
add x21,x19,x30 ## x21=i*4+j
slli x21,x21,2 ## [i][j]
add x22,x22,x11 ## &Q[i][k]
add x21,x21,x11 ## &Q[i][j]
flw f0,0(x21) ## f0=Q[i][j]
flw f1,0(x22) ## f1=Q[i][k]
flw f2,0(x20) ## f2=R[k][j]
fmul.s f0,f0,f1 ## Q[i][k]*Q[i][j]
fadd.s f2,f0,f2 ## R[k][j]+Q[i][k]*Q[i][j]
fsw f2,0(x20) ## R[k][j] += Q[i][k] * Q[i][j]
addi x16,x16,1
j lop5
end_lop5:
addi x16,x0,0
lop6: ## x20=& R[k][j]
bge x16,x13,end_lop6
sll x22,x16,x31 ## i*4
add x22,x22,x30 ## i*4+j
slli x22,x22,2 ## [i][j]
sll x21,x16,x31 ## i*4
add x21,x21,x15 ## i*4+k
slli x21,x21,2 ## [i][k]
add x22,x22,x11 ## &Q[i][j]
add x21,x21,x11 ## &Q[i][k]
flw f0,0(x21) ## Q[i][k]
flw f1,0(x22)## Q[i][j]
flw f2,0(x20) ## R[k][j]
fmul.s f2,f0,f2 ## R[k][j] * Q[i][k]
fsub.s f1,f1,f2 ## Q[i][j] - R[k][j] * Q[i][k]
fsw f1,0(x22) ## Q[i][j] -= R[k][j] * Q[i][k]
addi x16,x16,1
j lop6
end_lop6:
addi x30,x30,1
j lop4
end_lop4:
addi x15,x15,1
j lop1
end_lop1:
li a0,10
ecall




multiplication: ## 2x2 matrices
li x28,2
li x5,0
L1: li x6,0
L2: li x7,0
slli x30,x5,1
add x30,x30,x6
slli x30,x30,2
add x30,x10,x30
flw f0,0(x30)
L3: slli x29,x7,1
add x29,x29,x6
slli x29,x29,3
add x29,x12,x29
flw f1,0(x29)
slli x29,x5,1
add x29,x29,x7
slli x29,x29,2
add x29,x11,x29
flw f2,0(x29)
fmul.s f1,f2,f1
fadd.s f0,f0,f1
fsw  f0,0(x30)
addi x7,x7,1
bltu    x7,x28,L3
addi    x6,x6,1
bltu    x6,x28,L2
addi x5,x5,1
bltu x5,x28,L1
li a0,10
ecall

unrolled_modified_granny:
# Modified Gram-Schmidt orthogonalization in single precision
# Registers:
# x10: Base address of matrix A
# x11: Base address of matrix Q
# x12: Number of rows (M)
# x13: Number of columns (N)
# x17: q_k's
# x22: q_j's
## x15: k
# x16 :i
# x30: j
lopp1:
bge x15,x13,end_lopp1 
addi x16,x0,0
lopp2:
bge x16,x12,end_lopp2
slli x19,x16,2 ## i*4
add x19,x17,x19  ## & q_k[i]
slli x20,x16,2 ## i*4
add x20,x20,x15 ## i*4+k
slli x20,x20,2 ## [i][k]
add x20,x20,x11 ## &Q[i][k]
flw f0,0(x20)
fsw f0,0(x19)
addi x16,x16,1
j lopp2
end_lopp2:

## need to find norm(q_k) ,q_k in x17
addi x5,x10,0 ## x5=x10
addi x6,x31,0
addi x31,x11,0 ## x31=x11
addi x10,x17,0
addi x11,x12,0
jal norm
addi x10,x5,0
addi x11,x31,0
addi x31,x6,0
## now f0 holds the norm of q_k
la x7, zero_float  # Load address of zero_float (float 0.0) into x5
flw f1, 0(x7)      # Load 0.0f into f1 for comparison
feq.s x7, f0, f1   # Set x5 to 1 if f0 is equal to 0.0, otherwise set to 0

bnez x7, end_lopp3 # Branch if x5 is not zero, i.e., f0 is not zero

addi x16,x0,0
lopp3:
bge x16,x13,end_lopp3
slli x19,x16,2 ## i*4
add x19,x19,x15 ## i*4+k
slli x19,x19,2 ## [i][k]
add x20,x19,x11 ## x20=&Q[i][k]
flw f1,0(x20)  ## f1=Q[i][k]
fdiv.s f1,f1,f0 ## Q[i][k]/R[k][k]
fsw f1,0(x20) ## updated Q[i][k]
addi x16,x16,1
j lopp3
end_lopp3:
addi x30,x15,1 ## j=k+1
lopp4:
bge x30,x13,end_lopp4
slli x20,x15,2 ## k*4
add x20,x20,x30 ## k*4+j
slli x20,x20,2 ## [k][j]
add x20,x20,x14 ## &R[k][j]
la x7,zero_float
flw f5,0(x7)
fsw f5,0(x20) ## R[k][j]=0 , &R[k][j] in x20
## Q in x11, k in x15 and j in x30 , thats all what we need
## 0,4,8,12 for x19 , then we add to each

## First iteration that can be done in independent core
li x19,0 ## [0]
add x22,x19,x15 ## 0+k
slli x22,x22,2 ## [0][k] single precision
add x21,x19,x30 ## 0+j
slli x21,x21,2 ## [0][j]
add x22,x22,x11 ## &Q[0][k]
add x21,x21,x11 ## &Q[0][j]
flw f0,0(x22) ## f0=Q[0][k]
flw f1,0(x21) ## f1=Q[0][j]
flw f2,0(x20) ## f2=R[k][j]
fmul.s f0,f0,f1 ## Q[0][k]*Q[0][j]
fadd.s f2,f0,f2 ## R[k][j]+Q[0][k]*Q[0][j]
fsw f2,0(x20) ## R[k][j] += Q[0][k] * Q[0][j]
##

##Second iteration that can be done in independent core
li x19,4 ## [1]
add x22,x19,x15 ## 4+k
slli x22,x22,2 ## [1][k] single precision
add x21,x19,x30 ## 1+j
slli x21,x21,2 ## [1][j]
add x22,x22,x11 ## &Q[1][k]
add x21,x21,x11 ## &Q[1][j]
flw f0,0(x22) ## f0=Q[1][k]
flw f1,0(x21) ## f1=Q[1][j]
flw f2,0(x20) ## f2=R[k][j]
fmul.s f0,f0,f1 ## Q[1][k]*Q[1][j]
fadd.s f2,f0,f2 ## R[k][j]+Q[1][k]*Q[1][j]
fsw f2,0(x20) ## R[k][j] += Q[1][k] * Q[1][j]
##

##Third iteration that can be done in independent core
li x19,8 ## [2]
add x22,x19,x15 ## 8+k
slli x22,x22,2 ## [2][k] single precision
add x21,x19,x30 ## 8+j
slli x21,x21,2 ## [2][j]
add x22,x22,x11 ## &Q[2][k]
add x21,x21,x11 ## &Q[2][j]
flw f0,0(x22) ## f0=Q[2][k]
flw f1,0(x21) ## f1=Q[2][j]
flw f2,0(x20) ## f2=R[k][j]
fmul.s f0,f0,f1 ## Q[2][k]*Q[2][j]
fadd.s f2,f0,f2 ## R[k][j]+Q[2][k]*Q[2][j]
fsw f2,0(x20) ## R[k][j] += Q[2][k] * Q[2][j]
##

##Fourth iteration that can be done in independent core
li x19,12 ## [3]
add x22,x19,x15 ## 12+k
slli x22,x22,2 ## [3][k] single precision
add x21,x19,x30 ## 12+j
slli x21,x21,2 ## [3][j]
add x22,x22,x11 ## &Q[3][k]
add x21,x21,x11 ## &Q[3][j]
flw f0,0(x22) ## f0=Q[3][k]
flw f1,0(x21) ## f1=Q[3][j]
flw f2,0(x20) ## f2=R[k][j]
fmul.s f0,f0,f1 ## Q[3][k]*Q[3][j]
fadd.s f2,f0,f2 ## R[k][j]+Q[3][k]*Q[3][j]
fsw f2,0(x20) ## R[k][j] += Q[3][k] * Q[3][j]
##


#addi x16,x0,0 ## i=0
# lop5: ## x20=&R[k][j]
# bge x16,x13,end_lop5
# slli x19,x16,2 ## x19=i*4
# add x22,x19,x15 ## x20=i*4+k
# slli x22,x22,2 ## [i][k]
# add x21,x19,x30 ## x21=i*4+j
# slli x21,x21,2 ## [i][j]
# add x22,x22,x11 ## &Q[i][k]
# add x21,x21,x11 ## &Q[i][j]
# flw f0,0(x21) ## f0=Q[i][j]
# flw f1,0(x22) ## f1=Q[i][k]
# flw f2,0(x20) ## f2=R[k][j]
# fmul.s f0,f0,f1 ## Q[i][k]*Q[i][j]
# fadd.s f2,f0,f2 ## R[k][j]+Q[i][k]*Q[i][j]
# fsw f2,0(x20) ## R[k][j] += Q[i][k] * Q[i][j]
# addi x16,x16,1
#j lop5
#end_lop5:
addi x16,x0,0
lopp6: ## x20=& R[k][j]
bge x16,x13,end_lopp6
slli x22,x16,2 ## i*4
add x22,x22,x30 ## i*4+j
slli x22,x22,2 ## [i][j]
slli x21,x16,2 ## i*4
add x21,x21,x15 ## i*4+k
slli x21,x21,2 ## [i][k]
add x22,x22,x11 ## &Q[i][j]
add x21,x21,x11 ## &Q[i][k]
flw f0,0(x21) ## Q[i][k]
flw f1,0(x22)## Q[i][j]
flw f2,0(x20) ## R[k][j]
fmul.s f2,f0,f2 ## R[k][j] * Q[i][k]
fsub.s f1,f1,f2 ## Q[i][j] - R[k][j] * Q[i][k]
fsw f1,0(x22) ## Q[i][j] -= R[k][j] * Q[i][k]
addi x16,x16,1
j lopp6
end_lopp6:
addi x30,x30,1
j lopp4
end_lopp4:
addi x15,x15,1
j lopp1
end_lopp1:
li a0,10
ecall


unrolled_modified_granny2:
# Modified Gram-Schmidt orthogonalization in single precision
# Registers:
# x10: Base address of matrix A
# x11: Base address of matrix Q
# x12: Number of rows (M)
# x13: Number of columns (N)
# x17: q_k's
# x22: q_j's
## x15: k
# x16 :i
# x30: j

loppp1:
bge x15,x13,end_loppp1 
addi x16,x0,0
loppp2:
bge x16,x12,end_loppp2
slli x19,x16,2 ## i*4
add x19,x17,x19  ## & q_k[i]
slli x20,x16,2 ## i*4
add x20,x20,x15 ## i*4+k
slli x20,x20,2 ## [i][k]
add x20,x20,x11 ## &Q[i][k]
flw f0,0(x20)
fsw f0,0(x19)
addi x16,x16,1
j loppp2
end_loppp2:

## need to find norm(q_k) ,q_k in x17
addi x5,x10,0 ## x5=x10
addi x6,x31,0
addi x31,x11,0 ## x31=x11
addi x10,x17,0
addi x11,x12,0
jal norm
addi x10,x5,0
addi x11,x31,0
addi x31,x6,0
## now f0 holds the norm of q_k
la x7, zero_float  # Load address of zero_float (float 0.0) into x5
flw f1, 0(x7)      # Load 0.0f into f1 for comparison
feq.s x7, f0, f1   # Set x5 to 1 if f0 is equal to 0.0, otherwise set to 0

bnez x7, end_loppp3 # Branch if x5 is not zero, i.e., f0 is not zero
addi x10,x5,0
addi x11,x31,0

addi x16,x0,0
loppp3:
bge x16,x13,end_loppp3
slli x19,x16,2 ## i*4
add x19,x19,x15 ## i*4+k
slli x19,x19,2 ## [i][k]
add x20,x19,x11 ## x20=&Q[i][k]
flw f1,0(x20)  ## f1=Q[i][k]
fdiv.s f1,f1,f0 ## Q[i][k]/R[k][k]
fsw f1,0(x20) ## updated Q[i][k]
addi x16,x16,1
j loppp3
end_loppp3:
addi x30,x15,1 ## j=k+1
loppp4:
bge x30,x13,end_loppp4
slli x20,x15,2 ## k*4
add x20,x20,x30 ## k*4+j
slli x20,x20,2 ## [k][j]
add x20,x20,x14 ## &R[k][j]
la x7,zero_float
flw f5,0(x7)
fsw f5,0(x20) ## R[k][j]=0 , &R[k][j] in x20
## Q in x11, k in x15 and j in x30 , thats all what we need
## 0,4,8,12 for x19 , then we add to each

## First iteration that can be done in independent core
li x19,0 ## [0]
add x22,x19,x15 ## 0+k
slli x22,x22,2 ## [0][k] single precision
add x21,x19,x30 ## 0+j
slli x21,x21,2 ## [0][j]
add x22,x22,x11 ## &Q[0][k]
add x21,x21,x11 ## &Q[0][j]
flw f0,0(x22) ## f0=Q[0][k]
flw f1,0(x21) ## f1=Q[0][j]
flw f2,0(x20) ## f2=R[k][j]
fmul.s f0,f0,f1 ## Q[0][k]*Q[0][j]
fadd.s f2,f0,f2 ## R[k][j]+Q[0][k]*Q[0][j]
fsw f2,0(x20) ## R[k][j] += Q[0][k] * Q[0][j]
##

##Second iteration that can be done in independent core
li x19,4 ## [1]
add x22,x19,x15 ## 4+k
slli x22,x22,2 ## [1][k] single precision
add x21,x19,x30 ## 1+j
slli x21,x21,2 ## [1][j]
add x22,x22,x11 ## &Q[1][k]
add x21,x21,x11 ## &Q[1][j]
flw f0,0(x22) ## f0=Q[1][k]
flw f1,0(x21) ## f1=Q[1][j]
flw f2,0(x20) ## f2=R[k][j]
fmul.s f0,f0,f1 ## Q[1][k]*Q[1][j]
fadd.s f2,f0,f2 ## R[k][j]+Q[1][k]*Q[1][j]
fsw f2,0(x20) ## R[k][j] += Q[1][k] * Q[1][j]
##

##Third iteration that can be done in independent core
li x19,8 ## [2]
add x22,x19,x15 ## 8+k
slli x22,x22,2 ## [2][k] single precision
add x21,x19,x30 ## 8+j
slli x21,x21,2 ## [2][j]
add x22,x22,x11 ## &Q[2][k]
add x21,x21,x11 ## &Q[2][j]
flw f0,0(x22) ## f0=Q[2][k]
flw f1,0(x21) ## f1=Q[2][j]
flw f2,0(x20) ## f2=R[k][j]
fmul.s f0,f0,f1 ## Q[2][k]*Q[2][j]
fadd.s f2,f0,f2 ## R[k][j]+Q[2][k]*Q[2][j]
fsw f2,0(x20) ## R[k][j] += Q[2][k] * Q[2][j]
##

##Fourth iteration that can be done in independent core
li x19,12 ## [3]
add x22,x19,x15 ## 12+k
slli x22,x22,2 ## [3][k] single precision
add x21,x19,x30 ## 12+j
slli x21,x21,2 ## [3][j]
add x22,x22,x11 ## &Q[3][k]
add x21,x21,x11 ## &Q[3][j]
flw f0,0(x22) ## f0=Q[3][k]
flw f1,0(x21) ## f1=Q[3][j]
flw f2,0(x20) ## f2=R[k][j]
fmul.s f0,f0,f1 ## Q[3][k]*Q[3][j]
fadd.s f2,f0,f2 ## R[k][j]+Q[3][k]*Q[3][j]
fsw f2,0(x20) ## R[k][j] += Q[3][k] * Q[3][j]
##


#addi x16,x0,0 ## i=0
# lop5: ## x20=&R[k][j]
# bge x16,x13,end_lop5
# slli x19,x16,2 ## x19=i*4
# add x22,x19,x15 ## x20=i*4+k
# slli x22,x22,2 ## [i][k]
# add x21,x19,x30 ## x21=i*4+j
# slli x21,x21,2 ## [i][j]
# add x22,x22,x11 ## &Q[i][k]
# add x21,x21,x11 ## &Q[i][j]
# flw f0,0(x21) ## f0=Q[i][j]
# flw f1,0(x22) ## f1=Q[i][k]
# flw f2,0(x20) ## f2=R[k][j]
# fmul.s f0,f0,f1 ## Q[i][k]*Q[i][j]
# fadd.s f2,f0,f2 ## R[k][j]+Q[i][k]*Q[i][j]
# fsw f2,0(x20) ## R[k][j] += Q[i][k] * Q[i][j]
# addi x16,x16,1
#j lop5
#end_lop5:

#  &R[k][j] in x20, k in x15, j in x30
# 0,4,8,12 ( your eyes :)) the one i love the most

## First iteration that could be done on independent cores 
##, once R[k][j] is finalized above
li x22,0 ## i=0
add x22,x22,x30 ## i*0+j 
slli x22,x22,2 ## [0][j]
li x21,0
add x21,x21,x15 ## i*0+k
slli x21,x21,2 ## [0][k]
add x22,x22,x11 ## &Q[0][j]
add x21,x21,x11 ## & Q[0][k]
flw f0,0(x21) ## Q[0][k]
flw f1,0(x22)## Q[0][j]
flw f2,0(x20) ## R[k][j]
fmul.s f2,f0,f2 ## R[k][j] * Q[0][k]
fsub.s f1,f1,f2 ## Q[0][j] - R[k][j] * Q[0][k]
fsw f1,0(x22) ## Q[0][j] -= R[k][j] * Q[0][k]
##

## Second iteration that could be done on independent cores 
##, once R[k][j] is finalized above
li x22,4 ## i=1
add x22,x22,x30 ## 4+j 
slli x22,x22,2 ## [1][j]
li x21,4
add x21,x21,x15 ## 4+k
slli x21,x21,2 ## [1][k]
add x22,x22,x11 ## &Q[1][j]
add x21,x21,x11 ## & Q[1][k]
flw f0,0(x21) ## Q[1][k]
flw f1,0(x22)## Q[1][j]
flw f2,0(x20) ## R[k][j]
fmul.s f2,f0,f2 ## R[k][j] * Q[1][k]
fsub.s f1,f1,f2 ## Q[1][j] - R[k][j] * Q[1][k]
fsw f1,0(x22) ## Q[1][j] -= R[k][j] * Q[1][k]
##

## Third iteration that could be done on independent cores 
##, once R[k][j] is finalized above
li x22,8 ## i=2
add x22,x22,x30 ## 8+j 
slli x22,x22,2 ## [2][j]
li x21,8
add x21,x21,x15 ## 8+k
slli x21,x21,2 ## [2][k]
add x22,x22,x11 ## &Q[2][j]
add x21,x21,x11 ## & Q[2][k]
flw f0,0(x21) ## Q[2][k]
flw f1,0(x22)## Q[2][j]
flw f2,0(x20) ## R[k][j]
fmul.s f2,f0,f2 ## R[k][j] * Q[2][k]
fsub.s f1,f1,f2 ## Q[2][j] - R[k][j] * Q[2][k]
fsw f1,0(x22) ## Q[2][j] -= R[k][j] * Q[2][k]
##

## Fourth iteration that could be done on independent cores 
##, once R[k][j] is finalized above
li x22,12 ## i=3
add x22,x22,x30 ## 12+j 
slli x22,x22,2 ## [3][j]
li x21,12
add x21,x21,x15 ## 12+k
slli x21,x21,2 ## [3][k]
add x22,x22,x11 ## &Q[3][j]
add x21,x21,x11 ## & Q[3][k]
flw f0,0(x21) ## Q[3][k]
flw f1,0(x22)## Q[3][j]
flw f2,0(x20) ## R[k][j]
fmul.s f2,f0,f2 ## R[k][j] * Q[3][k]
fsub.s f1,f1,f2 ## Q[3][j] - R[k][j] * Q[3][k]
fsw f1,0(x22) ## Q[3][j] -= R[k][j] * Q[3][k]
##

# addi x16,x0,0
# loppp6: ## x20=& R[k][j]
# bge x16,x13,end_loppp6
# slli x22,x16,2 ## i*4
# add x22,x22,x30 ## i*4+j
# slli x22,x22,2 ## [i][j]
# slli x21,x16,2 ## i*4
# add x21,x21,x15 ## i*4+k
# slli x21,x21,2 ## [i][k]
# add x22,x22,x11 ## &Q[i][j]
# add x21,x21,x11 ## &Q[i][k]
# flw f0,0(x21) ## Q[i][k]
# flw f1,0(x22)## Q[i][j]
# flw f2,0(x20) ## R[k][j]
# fmul.s f2,f0,f2 ## R[k][j] * Q[i][k]
# fsub.s f1,f1,f2 ## Q[i][j] - R[k][j] * Q[i][k]
# fsw f1,0(x22) ## Q[i][j] -= R[k][j] * Q[i][k]
# addi x16,x16,1
# j loppp6
# end_loppp6:

addi x30,x30,1
j loppp4
end_loppp4:
addi x15,x15,1
j loppp1
end_loppp1:
li a0,10
ecall



logbase2: ## final value in x11, input in x14
addi sp,sp,-12
sw x10,0(sp)
sw x6,4(sp)
sw x1,8(sp)
li x10,1 ## inital , maybe its 2 so logbase2(2)=1
li x6,2 ## start with 2 

d1:
beq x6,x14,doned1
slli x6,x6,1
addi x10,x10,1
j d1
doned1:
addi x11,x10,0 
lw x10,0(sp)
lw x6,4(sp)
lw x1,8(sp)
jalr x0,0(x1)