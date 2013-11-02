/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */




#include "SimParams.h"

void cudaInit(int argc, char **argv);

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, struct cudaGraphicsResource **cuda_vbo_resource, int size);

void copyArrayToDevice(void* device, const void* host, int offset, int size);


void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);

void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);

void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);


void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


void setParameters(SimParams *hostParams);

void integrateSystem1(double *pos,
                     double *vel,
                     double *acc,
                     double deltaTime,
                     uint numParticles);

void integrateSystem2(
                     double *vel,
                     double *acc,
                     double deltaTime,
                     uint numParticles);

void integrateSystem1A(double *posA,
                     double *velA,
                     double *accA,
                     double deltaTime,
                     uint numParticles);

void calcHash(uint*  gridParticleHash,
              uint*  gridParticleIndex,
              double* pos,
              int    numParticles);

void reorderDataAndFindCellStart(uint*  cellStart, uint*  cellEnd, double* sortedPos, double* sortedPosA, double* sortedVel,
double* sortedVelA,
double* sortedAcc,
double* sortedAccA,
uint*  gridParticleHash,
uint*  gridParticleIndex,
double* oldPos,
double* oldPosA,
double* oldVel,
double* oldVelA,
double* oldAcc,
double* oldAccA,
uint   numParticles,
uint   numCells);

void collide(
             double* Acc,
             double* AccA,
             double* sortedPos,
             double* sortedPosA,
             double* sortedVel,
             double* sortedVelA,
             double* sortedAcc,
             double* sortedAccA,
             uint*  gridParticleIndex,
             uint*  cellStart,
             uint*  cellEnd,
             uint   numParticles,
             uint   numCells,
             double* m_dTangSpring,
             int* NUM_VEC,
             int* m_TANG_indice,
             double* m_dTangSpring_W,
             double* m_dForce_ij,
             double* m_dContact_ij,
             double* m_dForce_W,
             double* m_dContact_W);

//void collide_BOLA(
//             double* Acc,
//             double* AccA,
//             double* dPos,
//             double* dPosA,
//             double* dVel,
//             double* dVelA,
//             double* dAcc,
//             double* dAccA,
//             uint*  gridParticleIndex,
//             uint*  cellStart,
//             uint*  cellEnd,
//             uint   numParticles,
//             uint   numCells,
//             double* m_dTangSpring, double delta_time, int* NUM_VEC, int* m_TANG_indice, double* m_dTangSpring_W, double* m_dForce_ij,double* m_dContact_ij,double* m_dForce_W,double* m_dContact_W);


void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);





