.data
zero_float: .float 0.0
matrixA:    .float 1.0,2.0,3.0,4.0,0.0,1.0,1.5,0.5,1.0,0.0,2.5,2.0,1.0,0.0,2.0,0.0 # 32x32 floats (4*4*4 bytes)
matrixQ:    .float 1.0,2.0,3.0,4.0,0.0,1.0,1.5,0.5,1.0,0.0,2.5,2.0,1.0,0.0,2.0,0.0   # Space for the resulting orthogonal matrix Q
matrixR:    .float 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0  # Space for the resulting upper triangular matrix R
q_k:        .float 0.0,0.0,0.0,0.0
matrixC:    .float 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
matrixI:    .float 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
matrixD:    .float 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
matrixIdent: .float 1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0
q_j:        .float 0.0,0.0,0.0,0.0
y:          .float 3.0,3.0,4.0,4.0
y_prime:    .float 0.0,0.0,0.0,0.0
x:          .float 0.0,0.0,0.0,0.0
Q_t:    .float 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 


.text
## given y is in x8, H is in x24
main:
    la x10, matrixA     # Load address of matrixA into x10
    la x11, matrixQ      # Load address of matrixQ into x11
    la x14,matrixR
    la x25,y_prime
    la x20,matrixC
    la x19,matrixI
    la x30, matrixD
    la x31, matrixIdent
    la x23,Q_t
    la x28,x
    la x8,y
    la x17,q_k
    la x22,q_j
    li x12, 4      # Number of rows (32)
    li x13, 4 
    ##li x19,2         # Number of columns (32)
         # Loop back if not end of matrix
    call granny     # Call the Gram-Schmidt function
    ## x11 --> Q matrix , x14 --> R matrix
## multiplication takes , C in x10, A in x11 , B in x9
    addi x9,x14,0
    addi x29,x10,0 
    addi x10,x20,0
    call multiplication #Checking if QR=A
    addi x26, x10, 0
    addi x27, x29, 0
    call MeanSquareError #result in f3
    # Assuming you need to print the matrices or verify them, add your specific code here
    # This could include printing the elements of matrixQ and matrixR to a terminal or file
    ## input in x26 matrix A, output in x10
    addi x10,x23,0
    addi x26,x11,0 
    call transposeMatrix 
    ## Q_t * Q, Q_t -> x10, Q--> 11
    addi x5,x11,0
    addi x11,x10,0
    addi x9,x5,0
    addi x10,x19,0
    call multiplication #Checking in Q(T)Q=I
    addi x26, x10, 0
    addi x27, x31 , 0
    call MeanSquareError
    addi x9, x29, 0
    addi x10, x30, 0 
    call multiplication #Checking if Q(T)A= R
    addi x26, x10, 0
    addi x27, x14, 0
    call MeanSquareError
    li a0,10
    ecall






copy_matrices: ## A in x10, Q in x11
    addi sp, sp, -32
    sw x6, 0(sp)
    sw x5, 4(sp)
    sw x1, 8(sp)
    sw x7,12(sp)
    sw x28,16(sp)
    sw x9,20(sp)
    fsw f8,24(sp)

    li x5, 0  # i = 0
    ## suppose loop limit is in x19, loading in main  # Loop limit for i and j, since matrix is 32x32

loopd:
    beq x5,x12,exit_loopd
    li x6, 0  # j = 0
loope:
    beq x6,x13,exit_loope
    slli x28, x5, 2  # x28 = i * 32 (offset for ith row)
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
    lw x6, 0(sp)
    lw x5, 4(sp)
    lw x1, 8(sp)
    lw x7,12(sp)
    lw x28,16(sp)
    lw x9,20(sp)
    flw f8,24(sp)
    addi sp, sp, 32
    jalr x0,0(x1)



#----------------------------------------NORM--------------------------------------------------

# Calculates the norm of a vector in single precision
# Registers:
# x10: Base address of the vector
# x11: Size of the vector
# Result: f0 (Norm of the vector in single precision)

norm:
    addi sp, sp, -24
    fsw f2, 0(sp)  # Save f2 on the stack
    sw x12, 4(sp)  # Save x12 on the stack
    sw x13, 8(sp)  # Save x13 on the stack
    sw x14, 12(sp)  # Save x14 on the stack
    sw x1, 16(sp)  # Save x1 on the stack
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
    flw f2, 0(sp)  # Save f2 on the stack
    lw x12, 4(sp)  # Save x12 on the stack
    lw x13, 8(sp)  # Save x13 on the stack
    lw x14, 12(sp)  # Save x14 on the stack
    lw x1, 16(sp)  # Save x1 on the stack
    addi sp, sp, 24 # Adjust stack pointer back
    jalr x0,0(x1) # Return from function





#-------------------------------------------------------DOT--------------------------------------------

# Calculates the dot product of two vectors in single precision
# Registers:
# x10: Base address of vector1
# x11: Base address of vector2
# x12: Size of the vectors
# Result: f0 (Dot product in single precision)
dot:
    addi sp, sp, -32
    fsw f1, 0(sp)
    fsw f2, 4(sp)
    sw x13, 8(sp)
    sw x14, 12(sp)
    sw x15, 16(sp)
    sw x16, 20(sp)
    sw x1, 24(sp)
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
    flw f1, 0(sp)
    flw f2, 4(sp)
    lw x13, 8(sp)
    lw x14, 12(sp)
    lw x15, 16(sp)
    lw x16, 20(sp)
    lw x1, 24(sp)
    addi sp, sp, 32
    jalr x0,0(x1)



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
granny:
    addi sp,sp,-52
    sw x15,0(sp)
    sw x16,4(sp)
    sw x19,8(sp)
    sw x20,12(sp)
    sw x5,16(sp)
    sw x30,20(sp)
    sw x31,24(sp)
    sw x10,28(sp)
    fsw f0,32(sp)
    fsw f1,36(sp)
    fsw f2,40(sp)
    sw x7,44(sp)
    sw x1,48(sp)
    li x15, 0  # k = 0
loop1:
    bge x15, x12, end_loop1
    li x16, 0  # i = 0

loop2:
    bge x16, x13, end_loop2
    slli x19, x16, 2
    add x19, x17, x19  # &q[i]
    slli x20, x16, 2 # 
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
    slli x20, x16, 2
    add x20, x20, x30
    slli x20, x20, 2
    add x20, x20, x11  # &Q[i][j]
    flw f1, 0(x20)
    fsw f1, 0(x19)  # q_j[i] = Q[i][j]
    addi x16, x16, 1
    j loop4
end_loop4:
    addi x5, x10,0
    addi x31,x11,0
    addi x10, x22, 0
    addi x11, x17, 0
    jal dot  # Call dot product between q_j and q_k
    addi x10, x5, 0
    addi x11, x31, 0

    slli x20, x30, 2
    add x20, x15, x20
    slli x20, x20, 2
    add x20, x20, x14  # &R[j][k]
    fsw f0, 0(x20)  # R[j][k] = result of dot product
    addi x16, x0, 0
loop5:
    bge x16, x13, end_loop5
    slli x19, x16, 2
    add x19, x17, x19  # &q_k[i]
    slli x20, x16, 2
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
    addi x31, x11, 0
    addi x10, x17, 0
    addi x11, x12, 0
    jal norm  # Call norm of q_k
    addi x10,x5,0
    addi x11,x31,0

    slli x20, x15, 2
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
    slli x20, x16, 2
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
    lw x15,0(sp)
    lw x16,4(sp)
    lw x19,8(sp)
    lw x20,12(sp)
    lw x5,16(sp)
    lw x30,20(sp)
    lw x31,24(sp)
    lw x10,28(sp)
    flw f0,32(sp)
    flw f1,36(sp)
    flw f2,40(sp)
    lw x7,44(sp)
    lw x1,48(sp)
    addi sp,sp,52
    jalr x0,0(x1)
## multiply matrices , A*B, A--> x14, B-->15
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
addi sp,sp,-64       
sw x1,0(sp)
sw x15,4(sp)
sw x16,8(sp)
sw x19,12(sp)
sw x20,16(sp)
sw x5,20(sp)
sw x31,24(sp)
sw x10,28(sp)
sw x7,32(sp)
sw x30,36(sp)
sw x22,40(sp)
sw x21,44(sp)
fsw f0,48(sp)
fsw f1,52(sp)
fsw f2,56(sp)
fsw f5,60(sp)
addi x15,x0,0
lop1:
bge x15,x13,end_lop1 
addi x16,x0,0
lop2:
bge x16,x12,end_lop2
slli x19,x16,2 ## i*4
add x19,x17,x19  ## & q_k[i]
slli x20,x16,2 ## i*4
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
addi x31,x11,0 ## x31=x11
addi x10,x17,0
addi x11,x12,0
jal norm
slli x19,x15,2 ## k*4
add x19,x19,x15 ## k*4+k
slli x19,x19,2 ## [k][k]
add x19,x19,x14 ##R[K][K]
fsw f0,0(x19)

## now f0 holds the norm of q_k
la x7, zero_float  # Load address of zero_float (float 0.0) into x5
flw f1, 0(x7)      # Load 0.0f into f1 for comparison
feq.s x7, f0, f1   # Set x5 to 1 if f0 is equal to 0.0, otherwise set to 0

bnez x7, end_lop3 # Branch if x5 is not zero, i.e., f0 is not zero
addi x10,x5,0
addi x11,x31,0

addi x16,x0,0
lop3:
bge x16,x13,end_lop3
slli x19,x16,2 ## i*4
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
slli x20,x15,2 ## k*4
add x20,x20,x30 ## k*4+j
slli x20,x20,2 ## [k][j]
add x20,x20,x14 ## &R[k][j]
la x7,zero_float
flw f5,0(x7)
fsw f5,0(x20) ## R[k][j]=0
addi x16,x0,0 ## i=0
lop5: ## x20=&R[k][j]
bge x16,x13,end_lop5
slli x19,x16,2 ## x19=i*4
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
j lop6
end_lop6:
addi x30,x30,1
j lop4
end_lop4:
addi x15,x15,1
j lop1
end_lop1:
lw x1,0(sp)
lw x15,4(sp)
lw x16,8(sp)
lw x19,12(sp)
lw x20,16(sp)
lw x5,20(sp)
lw x31,24(sp)
lw x10,28(sp)
lw x7,32(sp)
lw x30,36(sp)
lw x22,40(sp)
lw x21,44(sp)
flw f0,48(sp)
flw f1,52(sp)
flw f2,56(sp)
flw f5,60(sp)
addi sp,sp,64
jalr x0,0(x1)


multiplication: ## 4x4 matrices
## x28,x5,x6,x7,x30,f0,x29,f1,f2,x7,x1
addi sp,sp,-48
sw x28,0(sp)
sw x5,4(sp)
sw x6,8(sp)
sw x7,12(sp)
sw x30,16(sp)
fsw f0,20(sp)
sw x29,24(sp)
fsw f1,28(sp)
fsw f2,32(sp)
sw x7,36(sp)
sw x1,40(sp)
li x28,4    #to change
li x5,0
L1: li x6,0
L2: li x7,0
slli x30,x5,2   #to change
add x30,x30,x6
slli x30,x30,2   #single precision *2
add x30,x10,x30
flw f0,0(x30)
L3: slli x29,x7,2
add x29,x29,x6
slli x29,x29,2       #must change to 2
add x29,x9,x29
flw f1,0(x29)  
slli x29,x5,2
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
lw x28,0(sp)
lw x5,4(sp)
lw x6,8(sp)
lw x7,12(sp)
lw x30,16(sp)
flw f0,20(sp) 
lw x29,24(sp)
flw f1,28(sp)
flw f2,32(sp)
lw x7,36(sp)
lw x1,40(sp)
addi sp,sp,48
jalr x0,0(x1)



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
addi sp,sp,-64
sw x1,0(sp)
sw x15,4(sp)
sw x16,8(sp)
sw x19,12(sp)
sw x20,16(sp)
sw x5,20(sp)
sw x31,24(sp)
sw x10,28(sp)
sw x7,32(sp)
sw x30,36(sp)
sw x22,40(sp)
sw x21,44(sp)
fsw f0,48(sp)
fsw f1,52(sp)
fsw f2,56(sp)
fsw f5,60(sp)
addi x15,x0,0
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
addi x31,x11,0 ## x31=x11
addi x10,x17,0
addi x11,x12,0
jal norm
addi x10,x5,0
addi x11,x31,0
## now f0 holds the norm of q_k
la x7, zero_float  # Load address of zero_float (float 0.0) into x5
flw f1, 0(x7)      # Load 0.0f into f1 for comparison
feq.s x7, f0, f1   # Set x5 to 1 if f0 is equal to 0.0, otherwise set to 0

bnez x7, end_lopp3 # Branch if x5 is not zero, i.e., f0 is not zero
addi x10,x5,0
addi x11,x31,0

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
lw x1,0(sp)
lw x15,4(sp)
lw x16,8(sp)
lw x19,12(sp)
lw x20,16(sp)
lw x5,20(sp)
lw x31,24(sp)
lw x10,28(sp)
lw x7,32(sp)
lw x30,36(sp)
lw x22,40(sp)
lw x21,44(sp)
flw f0,48(sp)
flw f1,52(sp)
flw f2,56(sp)
flw f5,60(sp)
addi sp,sp,64
jalr x0,0(x1)


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
addi sp,sp,-64
sw x1,0(sp)
sw x15,4(sp)
sw x16,8(sp)
sw x19,12(sp)
sw x20,16(sp)
sw x5,20(sp)
sw x31,24(sp)
sw x10,28(sp)
sw x7,32(sp)
sw x30,36(sp)
sw x22,40(sp)
sw x21,44(sp)
fsw f0,48(sp)
fsw f1,52(sp)
fsw f2,56(sp)
fsw f5,60(sp)
addi x15,x0,0
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
addi x31,x11,0 ## x31=x11
addi x10,x17,0
addi x11,x12,0
jal norm
addi x10,x5,0
addi x11,x31,0
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
lw x1,0(sp)
lw x15,4(sp)
lw x16,8(sp)
lw x19,12(sp)
lw x20,16(sp)
lw x5,20(sp)
lw x31,24(sp)
lw x10,28(sp)
lw x7,32(sp)
lw x30,36(sp)
lw x22,40(sp)
lw x21,44(sp)
flw f0,48(sp)
flw f1,52(sp)
flw f2,56(sp)
flw f5,60(sp)
addi sp,sp,64
jalr x0,0(x1)


## BONUS 

## y=Hx ## concept , i dont know , ka function yes i know , ka algorithm yes i know , concept wise? no

## 4x4
## result is a 4x1 vector in x11
## given Matrix is in x27 A
## given vector  us in x24 v
## x12 has nbr of rows, x13 has nbr of columns
## i is in x16, j in x30
## generate y_prime
multiplyMatrixVector:
addi sp,sp,-40
sw x1,0(sp)
sw x16,4(sp)
sw x30,8(sp)
sw x19,12(sp)
sw x20,16(sp)
sw x21,20(sp)
fsw f4,24(sp)
fsw f5,28(sp)
fsw f6,32(sp)
li x16,0
for1:
beq x16,x12,end_for1
li x30,0
for2:
beq x30,x13,end_for2
slli x19,x16,2 ## i*4
add x19,x19,x11 ## &result[i]
slli x20,x16,2 ## i*4
add x20,x20,x30 ## i*4+j
slli x20,x20,2 ## [i][j]
add x20,x20,x27 ## & A[i][j]
slli x21,x30,2 ## j*4
add x21,x21,x24 ## &v[j]
flw f4,0(x21) ##v[j]
flw f5,0(x20) ##  A[i][j]
flw f6,0(x19) ## result[i]
fmul.s f4,f4,f5 ## v[j]*A[i][j]
fadd.s f6,f4,f6 ## result[i] +matrix[i][j] * vec[j]
fsw f6,0(x19) ## result[i] =result[i] +matrix[i][j] * vec[j]
addi x30,x30,1
j for2
end_for2:
addi x16,x16,1
j for1
end_for1:
lw x1,0(sp)
lw x16,4(sp)
lw x30,8(sp)
lw x19,12(sp)
lw x20,16(sp)
lw x21,20(sp)
flw f4,24(sp)
flw f5,28(sp)
flw f6,32(sp)
addi sp,sp,40
jalr x0,0(x1)
 
## input in x26 matrix A, output in x10
## rows in x12, column in x13
## i in x16, j in x30
transposeMatrix:
addi sp,sp,-24
sw x1,0(sp)
sw x16,4(sp)
sw x30,8(sp)
sw x19,12(sp)
fsw f6,16(sp)
li x16,0
for_loop1:
beq x16,x12,end_for_loop1
li x30,0
for_loop2:
beq x30,x13,end_for_loop2
slli x19,x16,2 ## i*4
add x19,x19,x30 ## i*4+j
slli x19,x19,2 ## [i][j]
add x19,x26,x19 ## &A[i][j]
flw f6,0(x19) ## f6=A[i][j]
slli x19,x30,2 ## j*4
add x19,x19,x16 ## j*4+i
slli x19,x19,2 ## [j][i]
add x19,x19,x10 ## & A_t[j][i]
fsw f6,0(x19) ## trans[j][i] = matrix[i][j]
addi x30,x30,1
j for_loop2
end_for_loop2:
addi x16,x16,1
j for_loop1
end_for_loop1:
lw x1,0(sp)
lw x16,4(sp)
lw x30,8(sp)
lw x19,12(sp)
flw f6,16(sp)
addi sp,sp,24
jalr x0,0(x1)

## input R in x14, input y_prime in x18
## rows in x12
## output in x10
backwardSubstitution:
addi sp,sp,-56
sw x5,0(sp)
sw x7,4(sp)
sw x1,8(sp)
sw x16,12(sp)
sw x19,16(sp)
sw x20,20(sp)
sw x30,24(sp)
fsw f0,28(sp)
fsw f7,32(sp)
fsw f8,36(sp)
fsw f11,40(sp)
fsw f12,44(sp)
li x5,-1
addi x16,x12,-1 ## i=N-1
for_lop1:
beq x16,x5,end_for_lop1
slli x19,x16,2 
add x20,x19,x10 ##& x[i]
add x19,x19,x18 ## &y[i]
flw f7,0(x19) ##
fsw f7,0(x20) ## x[i]=y[i]
addi x30,x16,1 ## j=i+1
for_lop2: ## x20=&x[i]
beq x30,x12,end_for_lop2
slli x19,x16,2 ## i*4
add x19,x19,x30 ## i*4+j
slli x19,x19,2 ## [i][j]
add x19,x19,x14 ##& R[i][j]
flw f8,0(x19) ##  f8=R[i][j]
slli x19,x30,2 ## j*4
add x19,x19,x10 ## &x[j]
flw f9,0(x19) ## f9=x[j]
flw f0,0(x20) ## f10=x[i]
fmul.s f8,f8,f9 ## R[i][j] * x[j]
fsub.s f0,f0,f8 ##x[i] - R[i][j] * x[j]
fsw f0,0(x20) ## x[i] =x[i]- R[i][j] * x[j]
addi x30,x30,1
j for_lop2
end_for_lop2:
slli x19,x16,2 ## i*4
add x19,x19,x16 ## i*4+i
slli x19,x19,2 ## [i][i]
add x19,x19,x14 ## &R[i][i]
flw f12,0(x19)
la x7, zero_float  # Load address of zero_float (float 0.0) into x5
flw f11, 0(x7)      # Load 0.0f into f11 for comparison
feq.s x7, f12, f11   # Set x5 to 1 if f0 is equal to 0.0, otherwise set to 0
bnez x7, end_branch2 #
flw f0,0(x20) ## x[i]
fdiv.s f0,f0,f12 ## x[i]/R[i][i]
fsw f0,0(x20) ## x[i]=x[i]/R[i][i]
end_branch2:
addi x16,x16,-1
j for_lop1
end_for_lop1:
lw x5,0(sp)
lw x7,4(sp)
lw x1,8(sp)
lw x16,12(sp)
lw x19,16(sp)
lw x20,20(sp)
lw x30,24(sp)
flw f0,28(sp)
flw f7,32(sp)
flw f8,36(sp)
flw f11,40(sp)
flw f12,44(sp)
jalr x0,0(x1)

MeanSquareError:#output in f3 base address of A in x26 and base address of B in x27
addi sp, sp, -44
sw x24, 40(sp)
sw x23, 36, (sp)
fsw f4, 32(sp)
sw x28, 28(sp)
fsw f3, 24(sp)
sw x5, 20(sp)
sw x6, 16(sp)
sw x30, 12(sp)
sw x31, 8(sp)
fsw f0, 4(sp)
fsw f1, 0(sp)
li x28, 4
li x24, 16    
fcvt.s.w f4, x24     
li x23, 0
fcvt.s.w f3, x23     
li x5, 0  #initialize i
L4:li x6, 0  #initialize j
L5:slli x30, x5, 2 #offset
add x30,x30,x6 #offset
slli x30,x30,2 #offset
add x30, x26, x30 #address of A[i][j]
add x31, x27, x30 #address of B[i][j]
flw f0, 0(x30) #load A[i][j]
flw f1, 0(x31) #load B[i][j]
fsub.s f1, f0, f1
fmul.s f1, f1, f1
fadd.s f3, f3, f1
addi x6, x6, 1
bltu x6, x28, L5
addi x5, x5, 1
bltu x5, x28, L4
fdiv.s f3, f3, f4
flw f1, 0(sp)
flw f0, 4(sp)
lw x31, 8(sp)
lw x30, 12(sp)
lw x6, 16(sp)
lw x5, 20(sp)
flw f3, 24(sp)
lw x28, 28(sp)
flw f4, 32(sp)
lw x23, 36, (sp)
lw x24, 40(sp)
addi sp, sp, 44
jalr x0,0(x1)










## y=Hx


# Modified Gram-Schmidt orthogonalization in single precision
# Registers:F
# x10: Base address of matrix A
# x11: Base address of matrix Q
# x12: Number of rows (M)
# x13: Number of columns (N)
# x17: q_k's ## from memory
# x22: q_j's ## from memory
solveforx: ## given y is in x8, H is in x24
## QR DECOMPOSE H 
## where modified takes A in x10,
## and result it in x11
## and x14
addi x10,x24,0 ## A---> H
call modified_granny ## now x11--> Q, x14-> R
addi x10,x23,0 ## result in Q_t
addi x26,x11,0 ## input Q in x26
call transposeMatrix ## our result in x10
## auto auto tghyrt ma3a x23 , same address :)
## our y is in x8, our Q_t is in x10
addi x11,x25,0 ## now x11 holds new addres for y_prime
addi x24,x8,0 ## now x24--> holds y
addi x27,x10,0
call multiplyMatrixVector
## y_prime is in x11, R--> x14
addi x18,x11,0 ## y_prime --> x18
addi x10,x28,0
call backwardSubstitution  ## our result in x10
li a0,10
ecall