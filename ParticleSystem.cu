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

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"
//#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h


#include "ParticleSystem.cuh"
#include "ParticleSystem.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string.h>

#include <cutil_math.h>


#include <cutil_inline.h>

#include "thrust/device_ptr.h"

#include <thrust/tuple.h>
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"



#include "SimParams.h"

#include "particles_kernel.cu"


#include "config_system.h"
#include <algorithm>

using namespace std;
//#include "particles_kernel.cuh"

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>


// simulation parameters
//#include "config_system_geometry.h"
//#include "config_gas.h"

//#include "ParticleSystem.cu"

#define CUDART_PI_F         3.141592654f

#define VP 1.0

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize,double3 system_size,double3 system_origin, double timestep,
    bool bUseOpenGL) :
    m_bInitialized(false), m_bUseOpenGL(bUseOpenGL), m_numParticles(numParticles), m_hPos(0), m_hVel(0), m_hAcc(0),
    m_hPosA(0), m_hVelA(0), m_hAccA(0),
    m_hFxy(0), m_hContact(0),
    m_hParticleRadius(0),
    m_hParticleMass(0),
    m_gridSize(gridSize), m_timer(0), m_solverIterations(1)
{


  m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;


  double3 worldSize = system_size;

  // set simulation parameters
  m_params.gridSize = m_gridSize;
  m_params.numCells = m_numGridCells;
  m_params.numBodies = m_numParticles;



  //m_params.particleRadius = PARTICLE_RADIUS;
  //m_params.mass = DENSITY * (4.0 / 3.0) * CUDART_PI_F * m_params.particleRadius
  //    * m_params.particleRadius * m_params.particleRadius;
  //m_params.momentofinertia = (2.0 / 5.0) * m_params.mass
    //  * m_params.particleRadius * m_params.particleRadius;

  m_params.worldOrigin = system_origin;

  //TODO compute max particle radius

  double maxParticleRadius=0.3;

  double cellSize = maxParticleRadius * 2.0; // cell size equal to particle diameter

  m_params.cellSize = make_double3(cellSize, cellSize, cellSize);
  m_params.xmax = -3.0 * maxParticleRadius +(system_size.x-system_origin.x); // (-3.0*params.particleRadius +SYSTEM_SIZEX )
  m_params.xmin = 3.0 * maxParticleRadius +(system_origin.x); // (3.0*params.particleRadius  -SYSTEM_SIZEX )

  m_params.ymax = -3.0 * maxParticleRadius +(system_size.y-system_origin.y); // (-3.0*params.particleRadius +SYSTEM_SIZEY)
  m_params.ymin = 3.0 * maxParticleRadius +(system_origin.y); // (3.0*params.particleRadius  -SYSTEM_SIZEY)

  m_params.zmax = -3.0 * maxParticleRadius +(system_size.z-system_origin.z);
  m_params.zmin = 3.0 * maxParticleRadius +(system_origin.z);

  m_params.LX  = system_size.x;
  m_params.LY = system_size.y;
  m_params.LZ = system_size.z;

  m_params.L = max(system_size.x,max(system_size.y,system_size.z));

  m_params.pLX = maxParticleRadius+ (system_size.x-system_origin.x);
  m_params.pLY = system_size.y-system_origin.y;
  m_params.pLZ = system_size.z-system_origin.z;

  m_params.mu = MU * MU;
  m_params.muw = MU_WALL * MU_WALL;

  m_params.nmaxc = NUMAX_CONT;

  m_params.dt = timestep;
  m_params.dt_2 = timestep / 2.;

  m_params.MIN_D = 2.0 * maxParticleRadius;
  m_params.MIN_D2 = pow(2.0 * maxParticleRadius, 2);

  _initialize(numParticles);
}

ParticleSystem::~ParticleSystem()
{
  _finalize();
  m_numParticles = 0;
}

void  ParticleSystem::_initialize(int numParticles){

	assert(!m_bInitialized);


  // allocate host storage
	m_hPos = new double[m_numParticles * 4];
	m_hVel = new double[m_numParticles * 4];
	m_hAcc = new double[m_numParticles * 4];

	m_hPosA = new double[m_numParticles * 4];
	m_hVelA = new double[m_numParticles * 4];
	m_hAccA = new double[m_numParticles * 4];

	m_hFxy = new double[NUMAX_CONT * m_numParticles * 4];
	m_hContact = new double[NUMAX_CONT * m_numParticles * 4];
	m_hParticleRadius= new double[m_numParticles];
	m_hParticleMass =new double[m_numParticles];

  m_hFwall = new double[7 * m_numParticles * 4]; // sixs walls
  m_hContact_W = new double[7 * m_numParticles * 4];

  memset(m_hPos, 0, m_numParticles * 4 * sizeof(double));
  memset(m_hVel, 0, m_numParticles * 4 * sizeof(double));
  memset(m_hAcc, 0, m_numParticles * 4 * sizeof(double));

  memset(m_hPosA, 0, m_numParticles * 4 * sizeof(double));
  memset(m_hVelA, 0, m_numParticles * 4 * sizeof(double));
  memset(m_hAccA, 0, m_numParticles * 4 * sizeof(double));

  memset(m_hFxy, 0, NUMAX_CONT * m_numParticles * 4 * sizeof(double));
  memset(m_hContact, 0, NUMAX_CONT * m_numParticles * 4 * sizeof(double));

  memset(m_hFwall, 0, m_numParticles * 4 * sizeof(double));
  memset(m_hContact_W, 0, m_numParticles * 4 * sizeof(double));

  m_hCellStart = new uint[m_numGridCells];
  memset(m_hCellStart, 0, m_numGridCells * sizeof(uint));

  m_hCellEnd = new uint[m_numGridCells];
  memset(m_hCellEnd, 0, m_numGridCells * sizeof(uint));

  memset(m_hParticleRadius, 0, m_numParticles * sizeof(double));
  memset(m_hParticleMass, 0, m_numParticles  * sizeof(double));
  // allocate GPU data
  int memSize = sizeof(double) * 4 * m_numParticles;

  allocateArray((void**) &m_dPos, memSize);
  allocateArray((void**) &m_dVel, memSize);
  allocateArray((void**) &m_dAcc, memSize);

  allocateArray((void**) &m_dPosA, memSize);
  allocateArray((void**) &m_dVelA, memSize);
  allocateArray((void**) &m_dAccA, memSize);

  allocateArray((void**) &m_dSortedPos, memSize);
  allocateArray((void**) &m_dSortedVel, memSize);
  allocateArray((void**) &m_dSortedAcc, memSize);

  allocateArray((void**) &m_dSortedPosA, memSize);
  allocateArray((void**) &m_dSortedVelA, memSize);
  allocateArray((void**) &m_dSortedAccA, memSize);

  allocateArray((void**) &m_dTangSpring, NUMAX_CONT * sizeof(double) * 3 * m_numParticles);

  cudaMemset(m_dTangSpring, 0.0,NUMAX_CONT * sizeof(double) * 3 * m_numParticles);

  allocateArray((void**) &NUM_VEC,      NUMAX_CONT * sizeof(int) * 3 * m_numParticles);

  cudaMemset(NUM_VEC, 0.0, NUMAX_CONT * sizeof(int) * 3 * m_numParticles);

  allocateArray((void**) &m_TANG_indice,      NUMAX_CONT * sizeof(int) * 3 * m_numParticles);

  cudaMemset(m_TANG_indice, 0.0, NUMAX_CONT * sizeof(int) * 3 * m_numParticles);

  allocateArray((void**) &m_dForce_ij,      NUMAX_CONT * sizeof(double) * 4 * m_numParticles);
  cudaMemset(m_dForce_ij, 0.0,      NUMAX_CONT * sizeof(double) * 4 * m_numParticles);

  allocateArray((void**) &m_dCONTACT_ij,      NUMAX_CONT * sizeof(double) * 4 * m_numParticles);
  cudaMemset(m_dCONTACT_ij, 0.0,      NUMAX_CONT * sizeof(double) * 4 * m_numParticles);

  allocateArray((void**) &m_dForce_W, 7 * sizeof(double) * 4 * m_numParticles);

  cudaMemset(m_dForce_W, 0.0, 7 * sizeof(double) * 4 * m_numParticles);

  allocateArray((void**) &m_dCONTACT_W,7 * sizeof(double) * 4 * m_numParticles);

  cudaMemset(m_dCONTACT_W, 0.0, 7 * sizeof(double) * 4 * m_numParticles);

  allocateArray((void**) &m_dTangSpring_W,7 * sizeof(double) * 3 * m_numParticles);
  cudaMemset(m_dTangSpring_W, 0.0, 7 * sizeof(double) * 3 * m_numParticles);

  allocateArray((void**) &m_dGridParticleHash, m_numParticles * sizeof(uint));
  allocateArray((void**) &m_dGridParticleIndex, m_numParticles * sizeof(uint));

  allocateArray((void**) &m_dCellStart, m_numGridCells * sizeof(uint));
  allocateArray((void**) &m_dCellEnd, m_numGridCells * sizeof(uint));
  allocateArray((void**) &m_dR, m_numParticles*sizeof(double));
  allocateArray((void**) &m_dMass, m_numParticles*sizeof(double));
  //cutilCheckError(cutCreateTimer(&m_timer));

  m_updFrequency = 1000;
  m_iterations = 0;
  m_bInitialized = true;
}

void ParticleSystem::_finalize()
{
  assert(m_bInitialized);

  delete[] m_hPos;
  delete[] m_hVel;
  delete[] m_hAcc;

  delete[] m_hCellStart;
  delete[] m_hCellEnd;
  delete[] m_hParticleRadius;
  delete[] m_hParticleMass;

  cudaFree(m_dVel);
  cudaFree(m_dAcc);
  cudaFree(m_dSortedPos);
  cudaFree(m_dSortedVel);
  cudaFree(m_dR);
  cudaFree(m_dMass);
  freeArray(m_dGridParticleHash);
  freeArray(m_dGridParticleIndex);
  freeArray(m_dCellStart);
  freeArray(m_dCellEnd);
  freeArray(m_dTangSpring);
  freeArray(NUM_VEC);
  freeArray(m_TANG_indice);
}

// step the simulation
void ParticleSystem::update(double deltaTime)
{
//     assert(m_bInitialized);

//     double *dPos;
//     dPos = (double *) m_dPos;

  // integrate
#ifdef _DEBUG
  printf("Debug info: Integrator step started for %d particles\n", m_numParticles);
#endif
  integrateSystem1(m_dPos, m_dVel, m_dAcc, deltaTime, m_numParticles);

//	dumpParticles(0,m_numParticles);

#ifdef _DEBUG

  printf("Debug info: Integrator step finished\n");
#endif
//  // calculate grid hash
//  calcHash(m_dGridParticleHash, m_dGridParticleIndex, m_dPos, m_numParticles);
//
//  // sort particles based on hash
//  sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);
//
//  // reorder particle arrays into sorted order and
//  // find start and end of each cell
//  reorderDataAndFindCellStart(m_dCellStart, m_dCellEnd, m_dSortedPos,
//      m_dSortedPosA, m_dSortedVel, m_dSortedVelA, m_dSortedAcc, m_dSortedAccA,
//      m_dGridParticleHash, m_dGridParticleIndex, m_dPos, m_dPosA, m_dVel,
//      m_dVelA, m_dAcc, m_dAccA, m_numParticles, m_numGridCells);
//
//#ifdef _DEBUG
//  printf("Debug info: Colliding particles\n");
//#endif
//  // process collisions
//
//  collide(m_dAcc, m_dAccA, m_dSortedPos, m_dSortedPosA, m_dSortedVel,
//      m_dSortedVelA, m_dSortedAcc, m_dSortedAccA, m_dGridParticleIndex,
//      m_dCellStart, m_dCellEnd, m_numParticles, m_numGridCells, m_dTangSpring,
//      NUM_VEC, m_TANG_indice, m_dTangSpring_W, m_dForce_ij,
//      m_dCONTACT_ij, m_dForce_W, m_dCONTACT_W);
//
////  if (m_iterations > SHUT_TIME)
////    {
////      collide_BOLA(m_dAcc, m_dAccA, m_dPos, m_dPosA, m_dVel, m_dVelA, m_dAcc,
////          m_dAccA, m_dGridParticleIndex, m_dCellStart, m_dCellEnd,
////          m_numParticles, m_numGridCells, m_dTangSpring, deltaTime, NUM_VEC,
////          m_TANG_indice, m_dTangSpring_W, m_dForce_ij, m_dCONTACT_ij,
////          m_dForce_W, m_dCONTACT_W);
////    }
//  integrateSystem1A(m_dPosA, m_dVelA, m_dAccA, deltaTime, m_numParticles);
//
//
//  integrateSystem2(m_dVel, m_dAcc, deltaTime, m_numParticles);
//
////  // print out
////  if (m_iterations % m_updFrequency == 0 && m_iterations >= SHUT_TIME)
////    {
////
////      dumpParticlesToFile(0, m_numParticles);
////    }
  m_iterations++;

  // if ( (int)((m_iterations)/m_updFrequency) > 10 ) m_updFrequency *= 10;
}
void ParticleSystem::dumpGrid()
{
  // dump grid information
  copyArrayFromDevice(m_hCellStart, m_dCellStart, 0,
      sizeof(uint) * m_numGridCells);
  copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint) * m_numGridCells);
  uint maxCellSize = 0;
  for (uint i = 0; i < m_numGridCells; i++)
    {
      if (m_hCellStart[i] != 0xffffffff)
        {
          uint cellSize = m_hCellEnd[i] - m_hCellStart[i];
//            printf("cell: %d, %d particles\n", i, cellSize);
          if (cellSize > maxCellSize)
            maxCellSize = cellSize;
        }
    }
  printf("maximum particles per cell = %d\n", maxCellSize);
}

void ParticleSystem::dumpParticles(uint start, uint count)
{
  // debug
  cudaDeviceSynchronize();
  copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(double) * 4 * count);
  copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(double) * 4 * count);
  copyArrayFromDevice(m_hAcc, m_dAcc, 0, sizeof(double) * 4 * count);

  for (uint i = start; i < start + count; i++)
    {
//        printf("%d: ", i);
      printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i * 4 + 0],
          m_hPos[i * 4 + 1], m_hPos[i * 4 + 2], m_hPos[i * 4 + 3]);
      printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i * 4 + 0],
          m_hVel[i * 4 + 1], m_hVel[i * 4 + 2], m_hVel[i * 4 + 3]);
      printf("acc: (%.4f, %.4f, %.4f, %.4f)\n", m_hAcc[i * 4 + 0],
                m_hAcc[i * 4 + 1], m_hAcc[i * 4 + 2], m_hAcc[i * 4 + 3]);
    }
}

inline double frand(){

	return rand() / (double) RAND_MAX;
}

void ParticleSystem::dumpParticlesToFile(std::ofstream &os ,uint start, uint count)
{
  // debug
  copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(double) * 4 * count);
  copyArrayFromDevice(m_hAcc, m_dAcc, 0, sizeof(double) * 4 * count);
  copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(double) * 4 * count);
  copyArrayFromDevice(m_hParticleRadius, m_dR, 0, sizeof(double)  * count);
  copyArrayFromDevice(m_hParticleMass, m_dMass, 0, sizeof(double)  * count);
  /*

   char filename1[100];
   sprintf(filename1, "tmp%d.vtk",m_iterations+1000000);
   FILE* file1 = fopen(filename1, "w");
   fprintf(file1,"# vtk DataFile Version 2.0 \n");       // # vtk DataFile Version 2.0
   fprintf(file1,"Generated by raul and alvaro \n");      //   Generated by raul_alvaro
   fprintf(file1,"ASCII \n");
   fprintf(file1,"DATASET POLYDATA \n");
   fprintf(file1," POINTS %d double \n",count);
   for(uint i=start; i<start+count; i++) {
   fprintf(file1, " %f %f %f \n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2]);
   }
   fprintf(file1,"VERTICES %d %d \n",count,count*2);
   for(uint i=start; i<start+count; i++) {
   fprintf(file1, " %d %d \n",1,(int)m_hPos[i*4+3]);
   }
   fprintf(file1,"POINT_DATA %d\n",count);
   fprintf(file1,"SCALARS radius float 1 \n");
   fprintf(file1,"LOOKUP_TABLE default \n");
   for(uint i=start; i<start+count; i++) {
   fprintf(file1,"%f \n",m_hVel[i*4+3]);
   }

   fclose(file1); */

  double RR1 = 0.0, TT1 = 0.0;
  double flag;
  for (uint i = start; i < count; i++)
    {
      RR1 += (m_hVelA[i * 4 + 0] * m_hVelA[i * 4 + 0]
          + m_hVelA[i * 4 + 1] * m_hVelA[i * 4 + 1]
          + m_hVelA[i * 4 + 2] * m_hVelA[i * 4 + 2]) * m_params.momentofinertia
          / 2.0;
      TT1 += (m_hVel[i * 4 + 0] * m_hVel[i * 4 + 0]
          + m_hVel[i * 4 + 1] * m_hVel[i * 4 + 1]
          + m_hVel[i * 4 + 2] * m_hVel[i * 4 + 2]) * m_params.mass / 2.0;
      // printf("%d %e %e %e %e \n",i,TT1,RR1,m_params.momentofinertia,m_params.mass); fflush(0);
    }
  double K = TT1; // (RR1+TT1);
  double DD = pow(2. / 64., 2);
  double distx, disty, distz, Ep = 0, DIST2;
  /*for(uint i=0;i<count;i++){
   for(uint j=0;j<count;j++) {
   distx = m_hPos[i*4+0]-m_hPos[j*4+0];
   disty = m_hPos[i*4+1]-m_hPos[j*4+1];
   distz = m_hPos[i*4+2]-m_hPos[j*4+2];
   DIST2 = distx*distx + disty*disty + distz*distz;
   // if(i==0 && j==100) printf("%f %f %lf %lf \n",sqrt(DIST2),pow((2./64.-sqrt(DIST2)),2),m_hPos[i*4+1],m_hPos[j*4+1]);
   if(DIST2 < DD  && i!=j) {
   // if(i==0 && j==100) printf("%d %d %f %f \n",i,j,sqrt(DIST2),pow((2./64.-sqrt(DIST2)),2));
   Ep += pow((2./64.-sqrt(DIST2)),2);
   }
   }
   }*/

  Ep *= (COLLIDE_SPRING * 0.25);

  printf("%d %e %e %e %e %e  Compruebo \n", m_iterations, K, Ep,
      K / (double) count, Ep / (double) count, K / Ep);
  fflush(0);

  // if ( K/Ep < 0.5e-2) coolling();
//
//  if (m_iterations == SHUT_TIME)
//    {
//      //  if (m_iterations==0){
//      Prepare_shut();
//    }
  save_data();

}

double*
ParticleSystem::getArray(ParticleArray array)
{
  assert(m_bInitialized);

  double* hdata = 0;
  double* ddata = 0;
  struct cudaGraphicsResource *cuda_vbo_resource = 0;

  switch (array)
    {
  default:
  case POSITION:
    hdata = m_hPos;
    ddata = m_dPos;
    break;
  case VELOCITY:
    hdata = m_hVel;
    ddata = m_dVel;
    break;
  case ACCELERATION:
    hdata = m_hAcc;
    ddata = m_dAcc;
    break;
  case POSITIONA:
    hdata = m_hPosA;
    ddata = m_dPosA;
    break;
  case VELOCITYA:
    hdata = m_hVelA;
    ddata = m_dVelA;
    break;
  case ACCELERATIONA:
    hdata = m_hAccA;
    ddata = m_dAccA;
    break;

    }

  copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource,
      m_numParticles * 4 * sizeof(double));
  return hdata;
}

void ParticleSystem::setArray(ParticleArray array, const double* data, int start,
    int count)
{
  assert(m_bInitialized);

  switch (array)
    {
  default:
  case POSITION:
    copyArrayToDevice(m_dPos, data, start * 4 * sizeof(double), count * 4 * sizeof(double));
    break;
  case VELOCITY:
    copyArrayToDevice(m_dVel, data, start * 4 * sizeof(double),
        count * 4 * sizeof(double));
    break;
  case ACCELERATION:
    copyArrayToDevice(m_dAcc, data, start * 4 * sizeof(double),
        count * 4 * sizeof(double));
    break;
  case POSITIONA:
    copyArrayToDevice(m_dPosA, data, start * 4 * sizeof(double),
        count * 4 * sizeof(double));
    break;
  case VELOCITYA:
    copyArrayToDevice(m_dVelA, data, start * 4 * sizeof(double),
        count * 4 * sizeof(double));
    break;
  case ACCELERATIONA:
    copyArrayToDevice(m_dAccA, data, start * 4 * sizeof(double),
        count * 4 * sizeof(double));
    break;
  case MASS:
	  copyArrayToDevice(m_dMass, data, start * sizeof(double),
	          count  * sizeof(double));
	  break;
  case RADIUS:
	  copyArrayToDevice(m_dR, data, start * sizeof(double),
	          count  * sizeof(double));
	  break;
    }
}

//
//double round(double value)
//{
//  return floor(value + 0.5);
//}

/*
 void
 ParticleSystem::reset(ParticleConfig config)
 {
 m_hPos[0] = 0.0;
 m_hPos[1] =-1.0*m_params.particleRadius;
 m_hPos[2] = 0.0;
 m_hPos[3] = 0.0;

 m_hVel[0] = 0.0;
 m_hVel[1] = VP;
 m_hVel[2] = 0.0;
 m_hVel[3] = m_params.particleRadius;

 m_hAcc[0] = 0.0;
 m_hAcc[1] = 0.0;
 m_hAcc[2] = 0.0;
 m_hAcc[3] = 0.0;

 m_hPosA[0] = sqrt(2.)/2.;
 m_hPosA[1] = 0.0;
 m_hPosA[2] = sqrt(2.)/2.;
 m_hPosA[3] = 0.0;


 m_hVelA[0] = 0.0;
 m_hVelA[1] = 0.0;
 m_hVelA[2] = 0.0;
 m_hVelA[3] = MASS;

 m_hAccA[0] = 0.0;
 m_hAccA[1] = 0.0;
 m_hAccA[2] = 0.0;
 m_hAccA[3] = 1.0;


 m_hPos[4] = 0.0;
 m_hPos[5] = 1.0f*m_params.particleRadius;
 m_hPos[6] = 0.00f;
 m_hPos[7] = 1.0f;

 m_hVel[4] = 0.0;
 m_hVel[5] = -VP;
 m_hVel[6] = 0.0;
 m_hVel[7] = m_params.particleRadius;

 m_hAcc[4] = 0.0;
 m_hAcc[5] = 0.0;
 m_hAcc[6] = 0.0;
 m_hAcc[7] = 0.0;

 m_hPosA[4] = sqrt(2.)/2.;
 m_hPosA[5] = 0.0;
 m_hPosA[6] = sqrt(2.)/2.;
 m_hPosA[7] = 0.0;


 m_hVelA[4] = 0.0;
 m_hVelA[5] = 0.0;
 m_hVelA[6] = 0.0;
 m_hVelA[7] = MASS;


 m_hAccA[4] = 0.0;
 m_hAccA[5] = 0.0;
 m_hAccA[6] = 0.0;
 m_hAccA[7] = 1.0;




 setArray(POSITION, m_hPos, 0, m_numParticles);
 setArray(VELOCITY, m_hVel, 0, m_numParticles);
 setArray(ACCELERATION, m_hAcc, 0, m_numParticles);
 setArray(POSITIONA, m_hPosA, 0, m_numParticles);
 setArray(VELOCITYA, m_hVelA, 0, m_numParticles);
 setArray(ACCELERATIONA, m_hAccA, 0, m_numParticles);



 }
 */

/*
 void
 ParticleSystem::reset(ParticleConfig config)
 {

 uint numberOfPartInEachSideX = 42;     // with 0.5 max 21 // 1.0  // max 45
 uint numberOfPartInEachSideZ = 42;
 uint numberOfPartInEachSideY = m_numParticles/(numberOfPartInEachSideX*numberOfPartInEachSideZ);

 double SX = (CYLINDER_RADI-2.*PARTICLE_RADIUS)*sqrt(2.0)/2.;
 double SZ = (CYLINDER_RADI-2.*PARTICLE_RADIUS)*sqrt(2.0)/2.;

 double spacingx = (2.0*SX/(double)numberOfPartInEachSideX);
 double spacingz = (2.0*SZ/(double)numberOfPartInEachSideZ);
 double spacingy = (2.0/64.0);
 printf("%d %d %d\n", numberOfPartInEachSideX ,numberOfPartInEachSideZ ,numberOfPartInEachSideY);
 printf("%f %f %f %f %f \n", spacingx,spacingy,spacingz,(2./64.),SX);
 srand(1973);
 uint i = 0;
 for(uint y=1; y<=numberOfPartInEachSideY+1; y++) {
 for(uint z=0; z<numberOfPartInEachSideZ; z++) {
 for(uint x=0; x<numberOfPartInEachSideX; x++) {
 if (i < m_numParticles) {
 double point[6];
 point[0] = frand();
 point[1] = frand();
 point[2] = frand();

 point[3] = frand();
 point[4] = frand();
 point[5] = frand();

 m_hPos[i*4]  =   spacingx * x - SX  + spacingx/2.0;
 m_hPos[i*4+1] =  spacingy * y - SYSTEM_SIZE   + spacingy/2.0;
 m_hPos[i*4+2] =  spacingz * z - SZ  + spacingz/2.0;
 m_hPos[i*4+3] = i;

 m_hVel[i*4]   = (2.0 * (point[0] - 0.5))*1.0;

 m_hVel[i*4+1] = (2.0 * (point[1] - 0.5))*0.0;

 m_hVel[i*4+2] = (2.0 * (point[2] - 0.5))*1.0;

 m_hVel[i*4+3] = PARTICLE_RADIUS;

 m_hAcc[i*4]   = 0.0;
 m_hAcc[i*4+1] = 0.0;
 m_hAcc[i*4+2] = 0.0;
 m_hAcc[i*4+3] = 0.0;

 m_hPosA[i*4]   = sqrt(2.0)/2.0;
 m_hPosA[i*4+1] = 0.0;
 m_hPosA[i*4+2] = 0.0;
 m_hPosA[i*4+3] = sqrt(2.0)/2.0;

 m_hVelA[i*4] =   (2.0 * (point[3] - 0.5))*2.0;
 m_hVelA[i*4+1] = (2.0 * (point[4] - 0.5))*2.0;
 m_hVelA[i*4+2] = (2.0 * (point[5] - 0.5))*2.0;
 m_hVelA[i*4+3] = MASS;

 m_hAccA[i*4]   = 0.0;
 m_hAccA[i*4+1] = 0.0;
 m_hAccA[i*4+2] = 0.0;
 m_hAccA[i*4+3] = 1.0;                     //// LIM particles_kernel
 }
 i++;
 }
 }
 }

 printf("%d \n",i);

 setArray(POSITION, m_hPos, 0, m_numParticles);
 setArray(VELOCITY, m_hVel, 0, m_numParticles);
 setArray(ACCELERATION, m_hAcc, 0, m_numParticles);
 setArray(POSITIONA, m_hPosA, 0, m_numParticles);
 setArray(VELOCITYA, m_hVelA, 0, m_numParticles);
 setArray(ACCELERATIONA, m_hAccA, 0, m_numParticles);


 } */

void ParticleSystem::updateCardParams()
{
  // update constants
  setParameters(&m_params);
}

void ParticleSystem::save_data()
{

  copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(double) * 4 * m_numParticles);
  copyArrayFromDevice(m_hAcc, m_dAcc, 0, sizeof(double) * 4 * m_numParticles);
  //  copyArrayFromDevice(m_hPosA, m_dPosA, 0, sizeof(double)*4*m_numParticles);
  //  copyArrayFromDevice(m_hAccA, m_dAccA, 0, sizeof(double)*4*m_numParticles);
  copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(double) * 4 * m_numParticles);
  //  copyArrayFromDevice(m_hVelA, m_dVelA, 0, sizeof(double)*4*m_numParticles);

  char filename[100];
  sprintf(filename, "particles_%012d.txt", m_iterations);
  FILE* file = fopen(filename, "w");
  for (uint i = 0; i < m_numParticles; i++)
    {
      fprintf(file, "%d %.4f", i, m_params.particleRadius);
      fprintf(file, " %f %f %f %f", m_hPos[i * 4 + 0], m_hPos[i * 4 + 1],
          m_hPos[i * 4 + 2], m_hPos[i * 4 + 3]);
      fprintf(file, " %f %f %f %f", m_hVel[i * 4 + 0], m_hVel[i * 4 + 1],
          m_hVel[i * 4 + 2], m_hVel[i * 4 + 3]);
      fprintf(file, " %.9f %.9f %.9f %f", m_hAcc[i * 4 + 0], m_hAcc[i * 4 + 1],
          m_hAcc[i * 4 + 2], m_hAcc[i * 4 + 3]);

      //  fprintf(file, " %f %f %f %f", m_hPosA[i*4+0], m_hPosA[i*4+1], m_hPosA[i*4+2], m_hPosA[i*4+3]);
      //  fprintf(file, " %f %f %f %f", m_hVelA[i*4+0], m_hVelA[i*4+1], m_hVelA[i*4+2], m_hVelA[i*4+3]);
      //  fprintf(file, " %f %f %f %f", m_hAccA[i*4+0], m_hAccA[i*4+1], m_hAccA[i*4+2], m_hAccA[i*4+3]);
      fprintf(file, "\n");
    }
  fclose(file);

  copyArrayFromDevice(m_hFxy, m_dForce_ij, 0,NUMAX_CONT * sizeof(double) * 4 * m_numParticles);
  copyArrayFromDevice(m_hContact, m_dCONTACT_ij, 0,      NUMAX_CONT * sizeof(double) * 4 * m_numParticles);
  char filename2[100];
  sprintf(filename2, "contact_%012d.txt", m_iterations);
  FILE* file2 = fopen(filename2, "w");
  for (uint i = 0; i < NUMAX_CONT * m_numParticles; i++)
    {
      if (m_hFxy[i * 4 + 0] != 0 || m_hFxy[i * 4 + 1] != 0
          || m_hFxy[i * 4 + 2] != 0)
        {
          fprintf(file2, " %.12lf %.12lf %.12lf %.12lf", m_hFxy[i * 4 + 0],
              m_hFxy[i * 4 + 1], m_hFxy[i * 4 + 2], m_hFxy[i * 4 + 3]);
          fprintf(file2, " %.12lf %.12lf %.12lf %.12lf", m_hContact[i * 4 + 0],
              m_hContact[i * 4 + 1], m_hContact[i * 4 + 2],
              m_hContact[i * 4 + 3]);
          fprintf(file2, "\n");
        }
    }
  fclose(file2);

  /* copyArrayFromDevice(m_hFwall, m_dForce_W, 0, 7 *sizeof(double)*4*m_numParticles);
   copyArrayFromDevice(m_hContact_W,m_dCONTACT_W, 0, 7 *sizeof(double)*4*m_numParticles);
   char filename3[100];
   sprintf(filename3, "wallcontact_%012d.txt",m_iterations);
   FILE* file3 = fopen(filename3, "w");
   for(uint i=0; i<7*m_numParticles; i++) {
   if(m_hFwall[i*4+0]!=0 || m_hFwall[i*4+1]!=0 || m_hFwall[i*4+2]!=0){
   fprintf(file3, " %.12lf %.12lf %.12lf %.12lf", m_hFwall[i*4+0], m_hFwall[i*4+1], m_hFwall[i*4+2], m_hFwall[i*4+3]);
   fprintf(file3, " %.12lf %.12lf %.12lf %.12lf", m_hContact_W[i*4+0], m_hContact_W[i*4+1], m_hContact_W[i*4+2], m_hContact_W[i*4+3]);
   fprintf(file3, "\n");
   }
   }
   fclose(file3); */

}

//void ParticleSystem::Prepare_shut(){
//
//  copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(double) * 4 * m_numParticles);
//  copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(double) * 4 * m_numParticles);
//  copyArrayFromDevice(m_hVelA, m_dVelA, 0, sizeof(double) * 4 * m_numParticles);
//  copyArrayFromDevice(m_hAccA, m_dAccA, 0, sizeof(double) * 4 * m_numParticles);
//  copyArrayFromDevice(m_hAcc, m_dAcc, 0, sizeof(double) * 4 * m_numParticles);
//
//  int i_max = m_numParticles - 1;
//  uint i1 = 0;
//  double Y_MAX = m_hPos[0 * 4 + 1], tx, ty, tz, tw;
//  double R;
//
//  for (uint i = 1; i < m_numParticles; i++)
//    {
//      R = sqrt(
//          m_hPos[i * 4 + 2] * m_hPos[i * 4 + 2]
//              + m_hPos[i * 4 + 0] * m_hPos[i * 4 + 0]);
//      if (m_hPos[i * 4 + 1] > Y_MAX && R < 0.15)
//        {
//          i1 = i;
//          Y_MAX = m_hPos[i * 4 + 1];
//        }
//    }
//
//  tx = m_hPos[i1 * 4 + 0];
//  ty = m_hPos[i1 * 4 + 1];
//  tz = m_hPos[i1 * 4 + 2];
////     tw= m_hPos[i1*4+3];
//
//  m_hPos[i1 * 4 + 0] = m_hPos[i_max * 4 + 0];
//  m_hPos[i1 * 4 + 1] = m_hPos[i_max * 4 + 1];
//  m_hPos[i1 * 4 + 2] = m_hPos[i_max * 4 + 2];
//  //    m_hPos[i1*4+3] = m_hPos[i_max*4+3];
//
//  m_hPos[i_max * 4 + 0] = tx;
//  m_hPos[i_max * 4 + 1] = ty;
//  m_hPos[i_max * 4 + 2] = tz;
//  //   m_hPos[i_max*4+3] = tw;
//
//  m_hPos[i_max * 4 + 0] = 0.0;
//  m_hPos[i_max * 4 + 1] += 1.0 * PARTICLE_RADIUS; // put it really  out
//  m_hPos[i_max * 4 + 2] = 0.0;
//
//  m_hPos[i_max * 4 + 1] = m_hPos[i_max * 4 + 1] + R_BOLA - PARTICLE_RADIUS; // ready to shut
//  m_hVel[i_max * 4 + 3] = R_BOLA;
//  m_hVelA[i_max * 4 + 3] = MASS2;
//
//  m_hAcc[i_max * 4 + 0] = 0;
//  m_hAcc[i_max * 4 + 1] = 0;
//  m_hAcc[i_max * 4 + 2] = 0;
//
//  m_hAccA[i_max * 4 + 3] = 6.0;
//
//  printf("Start Shut %d \n", m_iterations);
//  setArray(POSITION, m_hPos, 0, m_numParticles);
//  setArray(VELOCITY, m_hVel, 0, m_numParticles);
//  setArray(VELOCITYA, m_hVelA, 0, m_numParticles);
//  setArray(ACCELERATION, m_hAcc, 0, m_numParticles);
//  setArray(ACCELERATIONA, m_hAccA, 0, m_numParticles);
//
//}

void
ParticleSystem::reset(ParticleConfig config)
{

  if (config == CONFIG_FROM_FILE)
    {

      char filename[100];
      sprintf(filename, "init.in");

      FILE* file = fopen(filename, "r");
      if (file)
        printf("The file %s has been opened and is currently been readed\n",
            filename);
      else
        printf("Input file can not be opened\n");

      int i;

      double radio, mass,P0, P1, P2, P3, V0, V1, V2, V3, A0, A1, A2, A3, PA0, PA1,
          PA2, PA3, VA0, VA1, VA2, VA3, AA0, AA1, AA2, AA3;

      //  while(fscanf(file,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", &i,&radio,&P0,&P1,&P2,&P3,&V0,&V1,&V2,&V3,&A0,&A1,&A2,&A3,&PA0,&PA1,&PA2,&PA3,&VA0,&VA1,&VA2,&VA3,&AA0,&AA1,&AA2,&AA3)!=EOF  && i<m_numParticles ) {
      while (fscanf(file,
          "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", &i,
          &radio, &mass,&P0, &P1, &P2,&P3, &V0, &V1, &V2, &V3, &A0, &A1, &A2, &A3)
          != EOF && i < m_numParticles)
        {

          m_hParticleRadius[i]=radio;
          m_hParticleMass[i] = mass;
          m_hPos[i * 4] = P0;
          m_hPos[i * 4 + 1] = P1;
          m_hPos[i * 4 + 2] = P2;
          m_hPos[i * 4 + 3] = P3;

          m_hVel[i * 4] = V0;
          m_hVel[i * 4 + 1] = V1;
          m_hVel[i * 4 + 2] = V2;
          m_hVel[i * 4 + 3] = V3;

          m_hAcc[i * 4] = A0;
          m_hAcc[i * 4 + 1] = A1;
          m_hAcc[i * 4 + 2] = A2;
          m_hAcc[i * 4 + 3] = 0.0;

          //intialize the angular position

          m_hPosA[i * 4 + 0] = sqrt(2.0) / 2.0;
          m_hPosA[i * 4 + 1] = 0.0;
          m_hPosA[i * 4 + 2] = 0.0;
          m_hPosA[i * 4 + 3] = sqrt(2.) / 2.0;

          //initialize the angular velocity

          m_hVelA[i * 4 + 0] = 0.0;
          m_hVelA[i * 4 + 1] = 0.0;
          m_hVelA[i * 4 + 2] = 0.0;
          m_hVelA[i * 4 + 3] = 0.0;

          m_hAccA[i * 4] = 0.0;
          m_hAccA[i * 4 + 1] = 0.0;
          m_hAccA[i * 4 + 2] = 0.0;
          m_hAccA[i * 4 + 3] = 1.0;
        }

      fclose(file);

      setArray(POSITION, m_hPos, 0, m_numParticles);
      setArray(VELOCITY, m_hVel, 0, m_numParticles);
      setArray(ACCELERATION, m_hAcc, 0, m_numParticles);
      setArray(POSITIONA, m_hPosA, 0, m_numParticles);
      setArray(VELOCITYA, m_hVelA, 0, m_numParticles);
      setArray(ACCELERATIONA, m_hAccA, 0, m_numParticles);
    }

}

//this function write the phase of the simulation to the file
//phase.dat to vi viewed with the mdvisualizer
void ParticleSystem::dump_phase(FILE *fout)
{
  //cudaDeviceSynchronize();


  copyArrayFromDevice(m_hPos, m_dPos, 0, 4 * sizeof(double) * m_numParticles);
  copyArrayFromDevice(m_hVel, m_dVel, 0, 4 * sizeof(double) * m_numParticles);
  copyArrayFromDevice(m_hAcc, m_dAcc, 0, 4 * sizeof(double) * m_numParticles);
  //TODO Uncomment this section DONE
  fprintf(fout, "#NewFrame\n");
  fprintf(fout, "#no_of_particles: %d\n", m_numParticles);
  fprintf(fout, "#compressed: no\n");
  fprintf(fout, "#gravity: 0.0 %lf 0.0\n", m_params.gravity.y);
//	    fprintf(f,"#time: %lf",
  fprintf(fout, "#timestep: %lf\n", m_params.dt);

  fprintf(fout, "#lx: %lf\n", 2 * m_params.LX);
  fprintf(fout, "#ly: %lf\n", 2 * m_params.LY);
  fprintf(fout, "#lz: %lf\n", 2 * m_params.LZ);
  fprintf(fout, "#x_0: %lf\n", getWorldOrigin().x);
  fprintf(fout, "#y_0: %lf\n", getWorldOrigin().y);
  fprintf(fout, "#z_0: %lf\n", getWorldOrigin().z);
  fprintf(fout, "#nprint: %d\n", m_updFrequency);

  fprintf(fout, "#EndOfHeader\n");

  for (unsigned int i = 0; i < m_numParticles; i++)
    {
//		fprintf(fout, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", m_hPos[i * 4 + 0], m_hPos[i * 4 + 1],
//				m_hPos[i * 4 + 2], PARTICLE_RADIUS, m_hVel[i * 4 + 0], m_hVel[i * 4 + 1],
//                                m_hVel[i * 4 + 2], m_hAcc[i * 4 + 0], m_hAcc[i * 4 + 1],
//                                m_hAcc[i * 4 + 2]);
      fprintf(fout, "%lf %lf %lf %lf \n", m_hPos[i * 4 + 0], m_hPos[i * 4 + 1],
          m_hPos[i * 4 + 2], m_hParticleRadius[i]);
    }
  fflush(fout);

}




  void   cudaInit(int argc, char **argv)
  {

    int count;
    cudaGetDeviceCount(&count);


    cudaDeviceProp deviceProp;
    int maxGPDeviceID=0; //id of device with max Gflops/s
    double maxGP=0.0;
    for(int i=0;i<count;i++){

        cudaGetDeviceProperties(&deviceProp,i);


    }



    cudaSetDevice(maxGPDeviceID);

  }

//  void   cudaGLInit(int argc, char **argv)
//  {
//    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
//    if (cutCheckCmdLineFlag(argc, (const char**) argv, "device"))
//      {
//        cutilDeviceInit(argc, argv);
//      }
//    else
//      {
//        cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
//      }
//  }

  void   allocateArray(void **devPtr, int size)
  {
    cudaMalloc(devPtr, size);
  }

  void freeArray(void *devPtr){
    cudaFree(devPtr);
  }

  void   threadSync()
  {
    cudaDeviceSynchronize();
  }

  void  copyArrayToDevice(void* device, const void* host, int offset, int size){

    cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice);
  }

//  void   registerGLBufferObject(uint vbo,      struct cudaGraphicsResource **cuda_vbo_resource){
//
//    cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
//  }

  void   unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
  {
    cudaGraphicsUnregisterResource(cuda_vbo_resource);
  }

  void *  mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource){
    void *ptr;
    cudaGraphicsMapResources(1, cuda_vbo_resource, 0);
    size_t num_bytes;

    cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, *cuda_vbo_resource);
    return ptr;
  }

  void   unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
  {
   cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
  }

  void   copyArrayFromDevice(void* host, const void* device,    struct cudaGraphicsResource **cuda_vbo_resource, int size){
    if (cuda_vbo_resource)
      device = mapGLBufferObject(cuda_vbo_resource);

    cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);

    if (cuda_vbo_resource)
      unmapGLBufferObject(*cuda_vbo_resource);
  }

  void   setParameters(SimParams *hostParams){
    // copy parameters to constant memory
     cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams));
  }

  //Round a / b to nearest higher integer value
  uint   iDivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
  }

  // compute grid and thread block size for a given number of elements
  void
  computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
  {
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
  }

  void
  integrateSystem1(double *pos, double *vel, double *acc, double deltaTime,
      uint numParticles)
  {

    //
    //    thrust::device_ptr<double4> d_pos4((double4 *)pos);
    //    thrust::device_ptr<double4> d_vel4((double4 *)vel);
    //    thrust::device_ptr<double4> d_acc4((double4 *)acc);
    //
    //
    //    try{
    //    thrust::for_each(
    //        thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4, d_acc4)),
    //        thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles, d_acc4+numParticles)),
    //        integrate_functor1(deltaTime));
    //    }
    //    catch (...)
    //    {
    //      printf("Exception thrown\n");
    //
    //    }

#ifdef _DEBUG
    printf("Debug info: Integrating system\n");
#endif
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    integrateSystem1D<<<numBlocks, numThreads>>>(deltaTime, pos, vel, acc,
        numParticles);

#ifdef _DEBUG

    printf("Debug info: System integrated\n");

#endif

  }

  void
  integrateSystem2(double *vel, double *acc, double deltaTime,
      uint numParticles)
  {
    //      thrust::device_ptr<double4> d_vel4((double4 *) vel);
    //      thrust::device_ptr<double4> d_acc4((double4 *) acc);
    //
    //      thrust::for_each(
    //                      thrust::make_zip_iterator(thrust::make_tuple(d_vel4, d_acc4)),
    //                      thrust::make_zip_iterator(
    //                                      thrust::make_tuple(d_vel4 + numParticles,
    //                                                      d_acc4 + numParticles)),
    //                      integrate_functor2(deltaTime));

    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);
#ifdef _DEBUG
    printf("Debug info: IntegrateSystem2D started\n");
#endif

    integrateSystem2D<<<numBlocks, numThreads>>>(deltaTime, vel, acc,
        numParticles);
#ifdef _DEBUG
    printf("Debug info: IntegrateSystem2D ended\n");

#endif


  }

  void
  integrateSystem1A(double *posA, double *velA, double *accA, double deltaTime,
      uint numParticles)
  {

    //    thrust::device_ptr<double4> d_posA4((double4 *)posA);
    //    thrust::device_ptr<double4> d_velA4((double4 *)velA);
    //    thrust::device_ptr<double4> d_accA4((double4 *)accA);
    //
    //    thrust::for_each(
    //        thrust::make_zip_iterator(thrust::make_tuple(d_posA4, d_velA4, d_accA4)),
    //        thrust::make_zip_iterator(thrust::make_tuple(d_posA4+numParticles, d_velA4+numParticles, d_accA4+numParticles)),
    //        integrate_functor1A(deltaTime));

    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);
#ifdef _DEBUG

    printf("Debug info: integrateSystem1AD started\n");
#endif
    integrateSystem1AD<<<numBlocks, numThreads>>>(deltaTime, posA, velA, accA,
        numParticles);
#ifdef _DEBUG
    printf("Debug info: integrateSystem1AD ended\n");
#endif

    cutilCheckMsg("Kernel execution failed");
  }

  void
  calcHash(uint* gridParticleHash, uint* gridParticleIndex, double* pos,
      int numParticles)
  {
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

#ifdef _DEBUG
    printf("Debug info:  Grid size for calcHash: (%d threads , %d blocks)\n",
        numThreads, numBlocks);
#endif

    // execute the kernel
    calcHashD<<<numBlocks, numThreads>>>(gridParticleHash, gridParticleIndex,
        pos, numParticles);

    // check if kernel invocation generated an error
    cutilCheckMsg("Kernel execution failed");
  }

  void
  reorderDataAndFindCellStart(uint* cellStart, uint* cellEnd, double* sortedPos,
      double* sortedPosA, double* sortedVel, double* sortedVelA,
      double* sortedAcc, double* sortedAccA, uint* gridParticleHash,
      uint* gridParticleIndex, double* oldPos, double* oldPosA, double* oldVel,
      double* oldVelA, double* oldAcc, double* oldAccA, uint numParticles,
      uint numCells)
  {
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    // set all cells to empty

#ifdef _DEBUG
    printf("Num cells: %d\n", numCells);
#endif
   cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint));

    uint smemSize = sizeof(uint) * (numThreads + 1);
    reorderDataAndFindCellStartD<<<numBlocks, numThreads, smemSize>>>(cellStart,
        cellEnd, (double4 *) sortedPos, (double4 *) sortedPosA,
        (double4 *) sortedVel, (double4 *) sortedVelA,
        (double4 *) sortedAcc, (double4 *) sortedAccA, gridParticleHash,
        gridParticleIndex, (double4 *) oldPos, (double4 *) oldPosA,
        (double4 *) oldVel, (double4 *) oldVelA, (double4 *) oldAcc,
        (double4 *) oldAccA, numParticles);
    cutilCheckMsg("Kernel execution failed");

  }
//
//  void   collide(double* Acc,
//		  double* AccA,
//		  double* sortedPos,
//		  double* sortedPosA,
//      double* sortedVel,
//      double* sortedVelA,
//      double* sortedAcc,
//      double* sortedAccA,
//      uint* gridParticleIndex,
//      uint* cellStart,
//      uint* cellEnd,
//      uint numParticles,
//      uint numCells,
//      double* m_dTangSpring,
//      int* NUM_VEC,
//      int* m_TANG_indice,
//      double* m_dTangSpring_W,
//      double* m_dForce_ij,
//      double* m_dContact_ij,
//      double* m_dForce_W,
//      double* m_dContact_W){
//#if USE_TEX
//    cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(double4)));
//    cutilSafeCall(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(double4)));
//    cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
//    cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
//#endif
//
//    // thread per particle
//    uint numThreads, numBlocks;
//    computeGridSize(numParticles, 256, numBlocks, numThreads);
//    //  printf("%d %d \n",numBlocks, numThreads);
//    // execute the kernel
//    collideD<<<numBlocks, numThreads>>>((double4*) Acc, (double4*) AccA,
//        (double4*) sortedPos, (double4*) sortedPosA, (double4*) sortedVel,
//        (double4*) sortedVelA, (double4*) sortedAcc, (double4*) sortedAccA,
//        gridParticleIndex, cellStart, cellEnd, numParticles,
//        (double3*) m_dTangSpring, (int*) NUM_VEC, (int*) m_TANG_indice,
//        (double4*) m_dForce_ij, (double4*) m_dContact_ij);
//
//#ifdef _DEBUG
//    printf("Debug info: CollideD ended\n");
//
//    printf("Debug info: CollideD1 started\n");
//#endif
//    collideD1<<<numBlocks, numThreads>>>((double4*) Acc, (double4*) AccA,
//        (double4*) sortedPos, (double4*) sortedPosA, (double4*) sortedVel,
//        (double4*) sortedVelA, (double4*) sortedAcc, (double4*) sortedAccA,
//        gridParticleIndex, cellStart, cellEnd, numParticles,
//        (double3*) m_dTangSpring_W, (double4*) m_dForce_W,
//        (double4*) m_dContact_W);
//#ifdef _DEBUG
//    printf("Debug info: CollideD1 ended\n");
//#endif
//    // check if kernel invocation generated an error
//    //cutilCheckMsg("Kernel execution failed");
//
//  }
//
//  void   collide_BOLA(double* Acc, double* AccA, double* dPos, double* dPosA,
//      double* dVel, double* dVelA, double* dAcc, double* dAccA,
//      uint* gridParticleIndex, uint* cellStart, uint* cellEnd,
//      uint numParticles, uint numCells, double* m_dTangSpring, int* NUM_VEC,
//      int* m_TANG_indice, double* m_dTangSpring_W, double* m_dForce_ij,
//      double* m_dContact_ij, double* m_dForce_W, double* m_dContact_W)
//  {
//#if USE_TEX
//    cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(double4)));
//    cutilSafeCall(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(double4)));
//    cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
//    cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
//#endif
//
//    // thread per particle
//    uint numThreads, numBlocks;
//    computeGridSize(numParticles, 256, numBlocks, numThreads);
//    //  printf("%d %d \n",numBlocks, numThreads);
//    // execute the kernel
//    collideD2<<<numBlocks, numThreads>>>((double4*) dPos, (double4*) dPosA,
//        (double4*) dVel, (double4*) dVelA, (double4*) dAcc,
//        (double4*) dAccA, gridParticleIndex, cellStart, cellEnd,
//        numParticles, (double3*) m_dTangSpring, (int*) NUM_VEC,
//        (int*) m_TANG_indice, (double4*) m_dForce_ij,
//        (double4*) m_dContact_ij);
//
//    // check if kernel invocation generated an error
//    //cutilCheckMsg("Kernel execution failed");
//
//  }

  void  sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles){

        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
        thrust::device_ptr<uint>(dGridParticleHash + numParticles),
        thrust::device_ptr<uint>(dGridParticleIndex));
  }


