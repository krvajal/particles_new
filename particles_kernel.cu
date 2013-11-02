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
/*
 * CUDA particle system kernel code.
 */
#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_
#include <stdio.h>
#include <math.h>
#include "math_constants.h"
#include "SimParams.h"
#include <thrust/tuple.h>
#include "double_util.h"
// simulation parameters
#include "config_system.h"
// simulation parameters in constant memory
__constant__ SimParams params;

//kernel function for the integration of the position of the particles

__global__ void integrateSystem1D(double deltaTime, double *posData,
		double *velData, double *accData, int numParticles) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < numParticles) {

		double3 pos = make_double3(posData[index * 4 + 0],
				posData[index * 4 + 1], posData[index * 4 + 2]);

		double3 vel = make_double3(velData[index * 4 + 0],
				velData[index * 4 + 1], velData[index * 4 + 2]);
		double3 acc = make_double3(accData[index * 4 + 0],
				accData[index * 4 + 1], accData[index * 4 + 2]);
		;

		vel += acc * (deltaTime / 2.0);

		pos += vel * deltaTime;

		// set this to zero to disable collisions with cube sides

		// store new values

		/* if (pos.x < -SYSTEM_SIZEX) { pos.x = pos.x + 2.0*SYSTEM_SIZEX; }
		 if (pos.x > SYSTEM_SIZEX)  { pos.x = pos.x - 2.0*SYSTEM_SIZEX; } */
		/* if (pos.y < -SYSTEM_SIZEY) { pos.y = pos.y + 2.0*SYSTEM_SIZEY; }
		 if (pos.y > SYSTEM_SIZEY)  { pos.y = pos.y - 2.0*SYSTEM_SIZEY; }
		 if (pos.z < -SYSTEM_SIZEZ) { pos.z = pos.z + 2.0*SYSTEM_SIZEZ; }
		 if (pos.z > SYSTEM_SIZEZ)  { pos.z = pos.z - 2.0*SYSTEM_SIZEZ; }*/

		if (pos.y > params.LY-params.Y0) {
			pos.y = params.LY-params.Y0;
			vel.y = -vel.y;
		}

		posData[index * 4 + 0] = pos.x;
		posData[index * 4 + 1] = pos.y;
		posData[index * 4 + 2] = pos.z;

		velData[index * 4 + 0] = vel.x;
		velData[index * 4 + 1] = vel.y;
		velData[index * 4 + 2] = vel.z;

	}

}

//kernel function to integrate the velocity of the particles over the time

__global__ void integrateSystem2D(double deltaTime, double *velData,
		double *accData, int numParticles) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < numParticles) {

		double3 vel = make_double3(velData[index * 4 + 0],
				velData[index * 4 + 1], velData[index * 4 + 2]);
		double3 acc = make_double3(accData[index * 4 + 0],
				accData[index * 4 + 1], accData[index * 4 + 2]);
		;

		vel += acc * (deltaTime / 2.0);

		// set this to zero to disable collisions with cube sides

		// store new values

		/* if (pos.x < -SYSTEM_SIZEX) { pos.x = pos.x + 2.0*SYSTEM_SIZEX; }
		 if (pos.x > SYSTEM_SIZEX)  { pos.x = pos.x - 2.0*SYSTEM_SIZEX; } */
		/* if (pos.y < -SYSTEM_SIZEY) { pos.y = pos.y + 2.0*SYSTEM_SIZEY; }
		 if (pos.y > SYSTEM_SIZEY)  { pos.y = pos.y - 2.0*SYSTEM_SIZEY; }
		 if (pos.z < -SYSTEM_SIZEZ) { pos.z = pos.z + 2.0*SYSTEM_SIZEZ; }
		 if (pos.z > SYSTEM_SIZEZ)  { pos.z = pos.z - 2.0*SYSTEM_SIZEZ; }*/

		velData[index * 4 + 0] = vel.x;
		velData[index * 4 + 1] = vel.y;
		velData[index * 4 + 2] = vel.z;

	}

}
__global__ void integrateSystem1AD(double deltaTime, double *posAData,
		double *velAData, double *accAData, int numParticles) {

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index > numParticles)
		return;

	double4 posADataParticle = make_double4(posAData[4 * index + 0],
			posAData[4 * index + 1], posAData[4 * index + 2],
			posAData[4 * index + 3]);
	double4 velADataParticle = make_double4(velAData[4 * index + 0],
			velAData[4 * index + 1], velAData[4 * index + 2],
			velAData[4 * index + 3]);

	double4 accADatapParticle = make_double4(accAData[4 * index + 0],
			accAData[4 * index + 1], accAData[4 * index + 2],
			accAData[4 * index + 3]);

	double4 posA = make_double4(posADataParticle.x, posADataParticle.y,
			posADataParticle.z, posADataParticle.w); // q(t)
	double3 velA = make_double3(velADataParticle.x, velADataParticle.y,
			velADataParticle.z); // w(t-dt/2)
	double3 velA1m2 = make_double3(velADataParticle.x, velADataParticle.y,
			velADataParticle.z); // w(t-dt/2)

	double3 accA = make_double3(accADatapParticle.x, accADatapParticle.y,
			accADatapParticle.z);

	velA += accA * (deltaTime / 2.0); // w(t)

	// new position = old position + velocity * deltaTime

	// set this to zero to disable collisions with cube sides

	double4 omega;
	omega.x = 0.0;
	omega.y = velA.x; // w(t)
	omega.z = velA.y;
	omega.w = velA.z;

	double Q[4][4] = { { posA.x, -posA.y, -posA.z, -posA.w }, { posA.y, posA.x,
			-posA.w, posA.z }, { posA.z, posA.w, posA.x, -posA.y }, { posA.w,
			-posA.z, posA.y, posA.x } };

	double4 posA_dot;
	/* posA_dot.x = Q[0][0]*omega.x + Q[0][1]*omega.y + Q[0][2]*omega.z + Q[0][3]*omega.w;
	 posA_dot.y = Q[1][0]*omega.x + Q[1][1]*omega.y + Q[1][2]*omega.z + Q[1][3]*omega.w;
	 posA_dot.z = Q[2][0]*omega.x + Q[2][1]*omega.y + Q[2][2]*omega.z + Q[2][3]*omega.w;
	 posA_dot.w = Q[3][0]*omega.x + Q[3][1]*omega.y + Q[3][2]*omega.z + Q[3][3]*omega.w; */
	//  because omega1.x = 0.0;
	posA_dot.x = Q[0][1] * omega.y + Q[0][2] * omega.z + Q[0][3] * omega.w;
	posA_dot.y = Q[1][1] * omega.y + Q[1][2] * omega.z + Q[1][3] * omega.w;
	posA_dot.z = Q[2][1] * omega.y + Q[2][2] * omega.z + Q[2][3] * omega.w;
	posA_dot.w = Q[3][1] * omega.y + Q[3][2] * omega.z + Q[3][3] * omega.w;
	posA_dot /= 2.0; // dot{q}(t)

	double4 q_t = posA; // q(t)
	posA += posA_dot * deltaTime / 2.0; // q(t+dt/2) = q(t) + dot{q}*dt/2.

	velA1m2 += accA * deltaTime; // w(t+dt/2)

	double4 omega1;
	omega1.x = 0.0;
	omega1.y = velA1m2.x; // w(t+dt/2)
	omega1.z = velA1m2.y;
	omega1.w = velA1m2.z;

	double Q1[4][4] = { { posA.x, -posA.y, -posA.z, -posA.w }, // Q(t+dt/2)
			{ posA.y, posA.x, -posA.w, posA.z }, { posA.z, posA.w, posA.x,
					-posA.y }, { posA.w, -posA.z, posA.y, posA.x } };

	double4 posA_dot1;

	/* posA_dot1.x = Q1[0][0]*omega1.x + Q1[0][1]*omega1.y + Q1[0][2]*omega1.z + Q1[0][3]*omega1.w;
	 posA_dot1.y = Q1[1][0]*omega1.x + Q1[1][1]*omega1.y + Q1[1][2]*omega1.z + Q1[1][3]*omega1.w;
	 posA_dot1.z = Q1[2][0]*omega1.x + Q1[2][1]*omega1.y + Q1[2][2]*omega1.z + Q1[2][3]*omega1.w;
	 posA_dot1.w = Q1[3][0]*omega1.x + Q1[3][1]*omega1.y + Q1[3][2]*omega1.z + Q1[3][3]*omega1.w; */
	//  because omega1.x = 0.0;
	posA_dot1.x = Q1[0][1] * omega1.y + Q1[0][2] * omega1.z
			+ Q1[0][3] * omega1.w;
	posA_dot1.y = Q1[1][1] * omega1.y + Q1[1][2] * omega1.z
			+ Q1[1][3] * omega1.w;
	posA_dot1.z = Q1[2][1] * omega1.y + Q1[2][2] * omega1.z
			+ Q1[2][3] * omega1.w;
	posA_dot1.w = Q1[3][1] * omega1.y + Q1[3][2] * omega1.z
			+ Q1[3][3] * omega1.w;

	posA_dot1 /= 2.0; // dot{q}(t+dt/2)

	posA = q_t + posA_dot1 * deltaTime; //  q(t+dt)  = q(t) + dot{q}(t+dt/2)
	posA = normalize(posA);

	// store new values

	posAData[index * 4 + 0] = posA.x;
	posAData[index * 4 + 1] = posA.y;
	posAData[index * 4 + 2] = posA.z;
	posAData[index * 4 + 3] = posA.w; // q(t+dt)
	velAData[index * 4 + 0] = velA1m2.x;
	velAData[index * 4 + 1] = velA1m2.y;
	velAData[index * 4 + 2] = velA1m2.z;
	velAData[index * 4 + 3] = velADataParticle.w; // w(t+dt/2)
	accAData[index * 4 + 0] = accA.x;
	accAData[index * 4 + 1] = accA.y;
	accAData[index * 4 + 2] = accA.z;
	accAData[index * 4 + 3] = accADatapParticle.w;

}

struct integrate_functor1
{
	double deltaTime;

	__host__ __device__
	integrate_functor1(double delta_time) :
			deltaTime(delta_time) {
	}

	template<typename Tuple>
	__host__ __device__

	void operator()(Tuple &t) {

		volatile double4 posData = thrust::get<0>(t);
		volatile double4 velData = thrust::get<1>(t);
		volatile double4 accData = thrust::get<2>(t);

		double3 pos = make_double3(posData.x, posData.y, posData.z);
		double3 vel = make_double3(velData.x, velData.y, velData.z);
		double3 acc = make_double3(accData.x, accData.y, accData.z);

		vel += acc * (deltaTime / 2.0);

//        printf("Estoy Aqui \n");

		pos += vel * deltaTime;

		// set this to zero to disable collisions with cube sides

		// store new values

		/* if (pos.x < -SYSTEM_SIZEX) { pos.x = pos.x + 2.0*SYSTEM_SIZEX; }
		 if (pos.x > SYSTEM_SIZEX)  { pos.x = pos.x - 2.0*SYSTEM_SIZEX; } */
		/* if (pos.y < -SYSTEM_SIZEY) { pos.y = pos.y + 2.0*SYSTEM_SIZEY; }
		 if (pos.y > SYSTEM_SIZEY)  { pos.y = pos.y - 2.0*SYSTEM_SIZEY; }
		 if (pos.z < -SYSTEM_SIZEZ) { pos.z = pos.z + 2.0*SYSTEM_SIZEZ; }
		 if (pos.z > SYSTEM_SIZEZ)  { pos.z = pos.z - 2.0*SYSTEM_SIZEZ; }*/

		if (pos.y > SYSTEMSIZEY) {
			pos.y = SYSTEMSIZEY;
			vel.y = -vel.y;
		}

		//printf("Test");
		thrust::get<0>(t) = make_double4(pos, posData.w);
		thrust::get<1>(t) = make_double4(vel, velData.w);
		thrust::get<2>(t) = make_double4(acc, accData.w);
	}
};

struct integrate_functor2
{
	double deltaTime;

	__host__ __device__
	integrate_functor2(double delta_time) :
			deltaTime(delta_time) {
	}

	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple t) {
		volatile double4 velData = thrust::get<0>(t);
		volatile double4 accData = thrust::get<1>(t);

		double3 vel = make_double3(velData.x, velData.y, velData.z);
		double3 acc = make_double3(accData.x, accData.y, accData.z);

		vel += acc * (deltaTime / 2.0);

		// store new values

		thrust::get<0>(t) = make_double4(vel, velData.w);
		thrust::get<1>(t) = make_double4(acc, accData.w);
	}
};

//integrate the angular position of the particles using a verlet integrator algorithm

struct integrate_functor1A
{
	double deltaTime;

	__host__ __device__
	integrate_functor1A(double delta_time) :
			deltaTime(delta_time) {
	}

	template<typename Tuple>
	__host__ __device__ void operator()(Tuple &t) {

		volatile double4 posAData = thrust::get<0>(t);
		volatile double4 velAData = thrust::get<1>(t);
		volatile double4 accAData = thrust::get<2>(t);

		double4 posA = make_double4(posAData.x, posAData.y, posAData.z,
				posAData.w); // q(t)
		double3 velA = make_double3(velAData.x, velAData.y, velAData.z); // w(t-dt/2)
		double3 velA1m2 = make_double3(velAData.x, velAData.y, velAData.z); // w(t-dt/2)

		double3 accA = make_double3(accAData.x, accAData.y, accAData.z);

		velA += accA * (deltaTime / 2.0); // w(t)

		// new position = old position + velocity * deltaTime

		// set this to zero to disable collisions with cube sides

		double4 omega;
		omega.x = 0.0;
		omega.y = velA.x; // w(t)
		omega.z = velA.y;
		omega.w = velA.z;

		double Q[4][4] = { { posA.x, -posA.y, -posA.z, -posA.w }, { posA.y,
				posA.x, -posA.w, posA.z }, { posA.z, posA.w, posA.x, -posA.y },
				{ posA.w, -posA.z, posA.y, posA.x } };

		double4 posA_dot;
		/* posA_dot.x = Q[0][0]*omega.x + Q[0][1]*omega.y + Q[0][2]*omega.z + Q[0][3]*omega.w;
		 posA_dot.y = Q[1][0]*omega.x + Q[1][1]*omega.y + Q[1][2]*omega.z + Q[1][3]*omega.w;
		 posA_dot.z = Q[2][0]*omega.x + Q[2][1]*omega.y + Q[2][2]*omega.z + Q[2][3]*omega.w;
		 posA_dot.w = Q[3][0]*omega.x + Q[3][1]*omega.y + Q[3][2]*omega.z + Q[3][3]*omega.w; */
		//  because omega1.x = 0.0;
		posA_dot.x = Q[0][1] * omega.y + Q[0][2] * omega.z + Q[0][3] * omega.w;
		posA_dot.y = Q[1][1] * omega.y + Q[1][2] * omega.z + Q[1][3] * omega.w;
		posA_dot.z = Q[2][1] * omega.y + Q[2][2] * omega.z + Q[2][3] * omega.w;
		posA_dot.w = Q[3][1] * omega.y + Q[3][2] * omega.z + Q[3][3] * omega.w;
		posA_dot /= 2.0; // dot{q}(t)

		double4 q_t = posA; // q(t)
		posA += posA_dot * deltaTime / 2.0; // q(t+dt/2) = q(t) + dot{q}*dt/2.

		velA1m2 += accA * deltaTime; // w(t+dt/2)

		double4 omega1;
		omega1.x = 0.0;
		omega1.y = velA1m2.x; // w(t+dt/2)
		omega1.z = velA1m2.y;
		omega1.w = velA1m2.z;

		double Q1[4][4] = { { posA.x, -posA.y, -posA.z, -posA.w }, // Q(t+dt/2)
				{ posA.y, posA.x, -posA.w, posA.z }, { posA.z, posA.w, posA.x,
						-posA.y }, { posA.w, -posA.z, posA.y, posA.x } };

		double4 posA_dot1;

		/* posA_dot1.x = Q1[0][0]*omega1.x + Q1[0][1]*omega1.y + Q1[0][2]*omega1.z + Q1[0][3]*omega1.w;
		 posA_dot1.y = Q1[1][0]*omega1.x + Q1[1][1]*omega1.y + Q1[1][2]*omega1.z + Q1[1][3]*omega1.w;
		 posA_dot1.z = Q1[2][0]*omega1.x + Q1[2][1]*omega1.y + Q1[2][2]*omega1.z + Q1[2][3]*omega1.w;
		 posA_dot1.w = Q1[3][0]*omega1.x + Q1[3][1]*omega1.y + Q1[3][2]*omega1.z + Q1[3][3]*omega1.w; */
		//  because omega1.x = 0.0;
		posA_dot1.x = Q1[0][1] * omega1.y + Q1[0][2] * omega1.z
				+ Q1[0][3] * omega1.w;
		posA_dot1.y = Q1[1][1] * omega1.y + Q1[1][2] * omega1.z
				+ Q1[1][3] * omega1.w;
		posA_dot1.z = Q1[2][1] * omega1.y + Q1[2][2] * omega1.z
				+ Q1[2][3] * omega1.w;
		posA_dot1.w = Q1[3][1] * omega1.y + Q1[3][2] * omega1.z
				+ Q1[3][3] * omega1.w;

		posA_dot1 /= 2.0; // dot{q}(t+dt/2)

		posA = q_t + posA_dot1 * deltaTime; //  q(t+dt)  = q(t) + dot{q}(t+dt/2)
		posA = normalize(posA);

		// store new values
		thrust::get<0>(t) = make_double4(posA.x, posA.y, posA.z, posA.w); // q(t+dt)
		thrust::get<1>(t) = make_double4(velA1m2, velAData.w); // w(t+dt/2)
		thrust::get<2>(t) = make_double4(accA, accAData.w);

	}
};

//*************************************************************
// calculate position in uniform grid
//*************************************************************

__device__ int3 calcGridPos(double3 p) {
	int3 gridPos;
	gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
	gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
	gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
	return gridPos;
}

//*************************************************************
// calculate address in grid from position (clamping to edges)
//*************************************************************
__device__ uint calcGridHash(int3 gridPos) {
	gridPos.x = gridPos.x & (params.gridSize.x - 1); // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (params.gridSize.y - 1);
	gridPos.z = gridPos.z & (params.gridSize.z - 1);
	return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x)
			+ __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint* gridParticleHash, // output
		uint* gridParticleIndex, // output
		double *pos, // input: positions
		uint numParticles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles)
		return;

	volatile double4 p = make_double4(pos[index * 4 + 0], pos[index * 4 + 1],
			pos[index * 4 + 2], pos[index * 4 + 3]);

	// get address in grid
	int3 gridPos = calcGridPos(make_double3(p.x, p.y, p.z));
	uint hash = calcGridHash(gridPos);

	// store grid hash and particle index
	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint* cellStart, // output: cell start index
		uint* cellEnd, // output: cell end index
		double4* sortedPos, // output: sorted positions
		double4* sortedPosA, // output: sorted positions
		double4* sortedVel, // output: sorted velocities
		double4* sortedVelA, // output: sorted velocities
		double4* sortedAcc, // output: sorted accelerations
		double4* sortedAccA, // output: sorted accelerations
		uint * gridParticleHash, // input: sorted grid hashes
		uint * gridParticleIndex, // input: sorted particle indices
		double4* oldPos, // input: sorted position array
		double4* oldPosA, // input: sorted position array
		double4* oldVel, // input: sorted velocity array
		double4* oldVelA, // input: sorted velocity array
		double4* oldAcc, // input: sorted accelerations array
		double4* oldAccA, // input: sorted accelerations array
		uint numParticles) {
	extern __shared__ uint sharedHash[]; // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;
	// handle case when no. of particles not multiple of block size
	if (index < numParticles) {
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0) {
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}

	__syncthreads();

	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x]) {
			cellStart[hash] = index;
			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1) {
			cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = gridParticleIndex[index];
		double4 pos = FETCH(oldPos, sortedIndex); // macro does either global read or texture fetch
		double4 vel = FETCH(oldVel, sortedIndex); // see particles_kernel.cuh
		double4 acc = FETCH(oldAcc, sortedIndex); // see particles_kernel.cuh

		double4 posA = FETCH(oldPosA, sortedIndex); // macro does either global read or texture fetch
		double4 velA = FETCH(oldVelA, sortedIndex); // see particles_kernel.cuh
		double4 accA = FETCH(oldAccA, sortedIndex); // see particles_kernel.cuh

		sortedPos[index] = pos;
		sortedVel[index] = vel;
		sortedAcc[index] = acc;

		sortedPosA[index] = posA;
		sortedVelA[index] = velA;
		sortedAccA[index] = accA;
	}

}

__device__
double distancePBC(double a, double b) {
	double delta = b - a;
	if (fabs(delta) >= SYSTEMSIZE) {
		if (a > b) {
			delta += 2.0 * SYSTEMSIZE;
		} else {
			delta -= 2.0 * SYSTEMSIZE;
		}
	}

	return delta;
}

__device__
double3 pointOfForceApplication(double3 posA, double3 posB) {
	double3 forceAppPoint;
	forceAppPoint.x = posA.x + distancePBC(posA.x, posB.x) / 2.0;
	forceAppPoint.y = posA.y + distancePBC(posA.y, posB.y) / 2.0;
	forceAppPoint.z = posA.z + distancePBC(posA.z, posB.z) / 2.0;

	return forceAppPoint;
}

// multiply double4 using the Fernando's method for multiplication of Quaternions
__device__
double4 multiplyQuaternions(double4 q1, double4 q2) {
	double4 q;
	double3 q3;

	double3 q13 = make_double3(q1.y, q1.z, q1.w);
	double3 q23 = make_double3(q2.y, q2.z, q2.w);
	;
	q3 = q1.x * q23 + q2.x * q13 + cross(q13, q23);

	q.x = q1.x * q2.x - dot(q13, q23);
	q.y = q3.x;
	q.z = q3.y;
	q.w = q3.z;

	return q;
}
// rotate function used to calculate the relative velocity between particles
__device__
double3 rotate(double4 quaternion, double3 omega) {
	double4 omega4 = make_double4(0.0, omega.x, omega.y, omega.z);
	double4 q_temp = multiplyQuaternions(quaternion, omega4);
	q_temp = multiplyQuaternions(q_temp,
			make_double4(quaternion.x, -quaternion.y, -quaternion.z,
					-quaternion.w));

	return make_double3(q_temp.y, q_temp.z, q_temp.w);
}

// collide two spheres using DEM method
//
//__device__
//double3 collideSpheres(double3 posA, double3 posB, double4 posAA, double4 posAB,
//		double3 velA, double3 velB, double3 velAA, double3 velAB,
//		double radiusA, double radiusB, double3* m_dTangSpring, uint pair,
//		uint N) {
//	double3 tangSpring;
//
//	tangSpring.x = m_dTangSpring[pair].x;
//	tangSpring.y = m_dTangSpring[pair].y;
//	tangSpring.z = m_dTangSpring[pair].z;
//
//	// double3 relPos = make_double3(posB.x-posA.x,distancePBC(posA.y, posB.y),posB.z-posA.z);
//	double3 relPos = make_double3(distancePBC(posA.x, posB.x),
//			distancePBC(posA.y, posB.y), distancePBC(posA.z, posB.z));
//	double dist2 = dot(relPos, relPos);
//	double3 force;
//	double min_d2 = MIND2;
//	double min_d = MIND;
//	if (radiusA != radiusB) {
//		min_d2 = MB2;
//		min_d = MB;
//	}
//
//	if (dist2 < min_d2) {
//
//		double dist = sqrt(dist2);
//		double3 norm = relPos / dist;
//		// force application point
//
//		double3 forceAppPoint = pointOfForceApplication(posA, posB);
//
//		// relative velocity
//
//		double3 posB_PBC = posA + relPos;
//
//		double3 relVel = 1.0
//				* ((velB - velA)
//						+ cross(rotate(posAB, velAB), forceAppPoint - posB_PBC)
//						- cross(rotate(posAA, velAA), forceAppPoint - posA));
//
//		// double3 normVel = dot(relVel, norm) * norm;
//		double normVelp = dot(relVel, norm);
//		// normal spring force
//		double overlap = min_d - dist;
//		double DA;
//		//  double Fnm = (-COLLIDE_SPRING*pow(overlap,1.5) + params.NormalDamping*normVelp);
//		double pow_overlap = pow(overlap, 0.5);
//		double Fnm = (-COLLIDE_SPRING * pow_overlap * overlap
//				+ BETA * sqrt(Sn * MASS / 2. * pow_overlap) * normVelp);
//
//		//double Fnm=1000.0;
//		double meff = MASS / 2;
//
////              double Fnm = (-1.0 * K_N * overlap + GAMMA_N * normVelp) * meff;
//
////                               printf("%f \n",overlap);
//
//		double3 fN = Fnm * norm;
//
//		//check if the force is attractive
//		//*********************************
//		if (dot(fN, norm) > 0.0)
//			fN = make_double3(0.0);
//
//		//*********************************
//		// double3 fNv = *normVel ;
//		// normal dashpot (damping) force
//
//		// fN += fNv;
//		force = fN;
//		// printf("%.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf \n",overlap/radiusA,fN.x,fN.y,fN.z,velA.x,velA.y,velA.z);
//
//		// relative tangential velocity
//		double3 normVel = normVelp * norm;
//		double3 tanVel = relVel - normVel;
//
//		tangSpring += tanVel * TIMESTEP;
//		double L_tangSA = dot(tangSpring, tangSpring); // length(tangSpring);
//		tangSpring -= dot(tangSpring, norm) * norm; // se carga la componente normal
//		double L1_tangSA = dot(tangSpring, tangSpring); // length(tangSpring);
//		if (L1_tangSA > 0.0)
//			tangSpring *= sqrt(L_tangSA / L1_tangSA);
//
//		double3 fT = params.shear * tangSpring;
//		fT += params.Tangentdamping * tanVel;
//
//		double L_ft = dot(fT, fT); // length(fT);
//		double L_fn = MU_WALL * dot(fN, fN); // length(fN);
//
//		if (L_ft > L_fn) {
//			fT = fT * sqrt(L_fn / L_ft);
//			tangSpring = (fT - params.Tangentdamping * tanVel) / params.shear;
//		}
//		force += fT;
//
//	} else {
//		tangSpring.x = 0.0;
//		tangSpring.y = 0.0;
//		tangSpring.z = 0.0;
//		force = make_double3(0.0);
//	}
//
//	m_dTangSpring[pair].x = tangSpring.x;
//	m_dTangSpring[pair].y = tangSpring.y;
//	m_dTangSpring[pair].z = tangSpring.z;
//
//	return force;
//}

//__device__
//double3 collideSpheres_wall(double3 posA, double3 posB, double4 posAA,
//		double4 posAB, double3 velA, double3 velB, double3 velAA, double3 velAB,
//		double radiusA, double radiusB, double3* m_dTangSpring_W, int pair) {
//
//	double3 tangSpring;
//
//	tangSpring.x = m_dTangSpring_W[pair].x;
//	tangSpring.y = m_dTangSpring_W[pair].y;
//	tangSpring.z = m_dTangSpring_W[pair].z;
//
//	// double3 relPos = make_double3(distancePBC(posA.x, posB.x),distancePBC(posA.y, posB.y),distancePBC(posA.z, posB.z));
//	// double3 relPos = make_double3(posB.x-posA.x,distancePBC(posA.y, posB.y),posB.z-posA.z);
//	double3 relPos = make_double3(distancePBC(posA.x, posB.x),
//			distancePBC(posA.y, posB.y), distancePBC(posA.z, posB.z));
//
//	double dist2 = dot(relPos, relPos);
//	double3 force;
//
//	if (dist2 < MIND2) {
//		double dist = sqrt(dist2);
//		double3 norm = relPos / dist;
//
//		// force application point
//		double3 forceAppPoint = pointOfForceApplication(posA, posB);
//
//		double3 relVel = (velB - velA)
//				+ cross(rotate(posAB, velAB), forceAppPoint - posB)
//				- cross(rotate(posAA, velAA), forceAppPoint - posA);
//
//		// double3 normVel = dot(relVel, norm) * norm;
//
//		double normVelp = dot(relVel, norm);
//		// normal spring force
//		double overlap = MIND - dist;
//		double pow_overlap = pow(overlap, 0.5);
//
//		double Fnm = (-COLLIDE_SPRING * pow_overlap * overlap
//				+ BETA * sqrt(Sn * MASS / 2. * pow_overlap) * normVelp);
//
//		double3 fN = Fnm * norm;
//		// double3 fNv = *normVel ;
//		// normal dashpot (damping) force
//
//		// fN += fNv;
//		force = fN;
//
//		// relative tangential velocity
//		double3 normVel = normVelp * norm;
//		double3 tanVel = relVel - normVel;
//
//		tangSpring += tanVel * TIMESTEP;
//		double L_tangSA = dot(tangSpring, tangSpring); // length(tangSpring);
//		tangSpring -= dot(tangSpring, norm) * norm; // se carga la componente normal
//		double L1_tangSA = dot(tangSpring, tangSpring); // length(tangSpring);
//		if (L1_tangSA > 0.0)
//			tangSpring *= sqrt(L_tangSA / L1_tangSA);
//
//		double3 fT = params.shear * tangSpring;
//		fT += params.Tangentdamping * tanVel;
//
//		double L_ft = dot(fT, fT); // length(fT);
//		double L_fn = MU_WALL2 * dot(fN, fN); // length(fN);
//
//		if (L_ft > L_fn) {
//			fT = fT * sqrt(L_fn / L_ft);
//			tangSpring = (fT - params.Tangentdamping * tanVel) / params.shear;
//		}
//		force += fT;
//
//	} else {
//		tangSpring.x = 0.0;
//		tangSpring.y = 0.0;
//		tangSpring.z = 0.0;
//		force = make_double3(0.0);
//	}
//
//	m_dTangSpring_W[pair].x = tangSpring.x;
//	m_dTangSpring_W[pair].y = tangSpring.y;
//	m_dTangSpring_W[pair].z = tangSpring.z;
//
//	return force;
//
//}
//
//// collide a particle against all other particles in a given cell
//__device__
//double3 collideCell(int3 gridPos, uint index, uint indice_i, double3 pos,
//		double3 vel, double4 posA, double3 velA, double4* oldPos,
//		double4* oldVel, double4* oldPosA, double4* oldVelA, uint* cellStart,
//		uint* cellEnd, double4* torque, double3* m_dTangSpring,
//		uint numParticles, int* NUM_VEC, int* m_TANG_indice,
//		double4* m_dForce_ij, double4* m_dContact_ij) {
//
//	uint gridHash = calcGridHash(gridPos);
//
//	// get start of bucket for this cell
//	uint startIndex = FETCH(cellStart, gridHash);
//
//	double3 force = make_double3(0.0);
//	double3 partToPartForce;
//	double3 partToPartTorque = make_double3(0.0);
//
//	uint pair;
//
//	if (startIndex != 0xffffffff) { // cell is not empty
//		// iterate over particles in this cell
//		uint endIndex = FETCH(cellEnd, gridHash);
//		for(uint j=startIndex; j<endIndex; j++) {
//			if (j != index) { // check not colliding with self
//
//					double3 pos2 = make_double3(FETCH(oldPos, j));
//					double R1 = FETCH(oldVel, index).w;
//					double R2 = FETCH(oldVel, j).w;
//
//					double3 vel2 = make_double3(FETCH(oldVel, j));
//					double4 posA2 = FETCH(oldPosA, j);
//					double3 velA2 = make_double3(FETCH(oldVelA, j));
//
//					uint vecino = FETCH(oldPos, j).w;
//					int indice_vecino;
//					uint row = indice_i*NUMAX_CONT;
//
//					uint found=0;
//					uint p=0;
//
//					while(p<NUM_VEC[indice_i] && found==0) {
//						if (m_TANG_indice[row + p]==vecino) {
//							// indice_vecino = p;
//							// pair=(row + indice_vecino);
//							pair = row + p;
//							found = 1;
//						}
//
//						p++;
//					}
//					if(found==0) {
//						// indice_vecino = NUM_VEC[indice_i];
//						// pair=(row + indice_vecino);
//						pair = row + NUM_VEC[indice_i];
//						m_TANG_indice[pair] = vecino;
//						m_dTangSpring[pair] = make_double3(0.0);
//						NUM_VEC[indice_i]++;
//					}
//
//					double3 forceAppPoint = pointOfForceApplication(pos, pos2);
//
//					// collide two spheres
//
//					partToPartForce = collideSpheres(pos, pos2, posA, posA2,vel, vel2, velA, velA2, R1, R2, m_dTangSpring, pair, numParticles*numParticles);
//					force += partToPartForce;
//
//					m_dForce_ij[pair].x = partToPartForce.x;
//					m_dForce_ij[pair].y = partToPartForce.y;
//					m_dForce_ij[pair].z = partToPartForce.z;
//					m_dForce_ij[pair].w = (double)indice_i;
//
//					m_dContact_ij[pair].x = forceAppPoint.x;
//					m_dContact_ij[pair].y = forceAppPoint.y;
//					m_dContact_ij[pair].z = forceAppPoint.z;
//					m_dContact_ij[pair].w = (double)vecino;
//
//					partToPartTorque = cross(forceAppPoint - pos,partToPartForce);
//					double4 PosAtemp= make_double4(posA.x,-posA.y,-posA.z,-posA.w);
//					partToPartTorque = rotate(PosAtemp,partToPartTorque);
//
//					torque[indice_i].x += partToPartTorque.x;
//					torque[indice_i].y += partToPartTorque.y;
//					torque[indice_i].z += partToPartTorque.z;
//					// torque[indice_i].w = 0.0 ;
//
//					if (m_dTangSpring[pair].x==0.0 && m_dTangSpring[pair].y==0.0 && m_dTangSpring[pair].z==0.0) {
//						NUM_VEC[indice_i]--;
//						if(NUM_VEC[indice_i]!=0) {
//							uint pair_to_move = row + NUM_VEC[indice_i];
//							m_TANG_indice[pair] = m_TANG_indice[pair_to_move];
//							m_dTangSpring[pair] = m_dTangSpring[pair_to_move];
//						}
//					}
//				}
//
//			}
//		}
//
//	return force;
//}
//
//// collide a particle against all other particles in a given cell
//__device__
//double3 collide_BOLA(uint indice_i, double3 pos, double3 vel, double4 posA,
//		double3 velA, double4* dPos, double4* dVel, double4* dPosA,
//		double4* dVelA, uint* cellStart, uint* cellEnd, double4* torque,
//		double3* m_dTangSpring, uint numParticles, int* NUM_VEC,
//		int* m_TANG_indice, double4* m_dForce_ij, double4* m_dContact_ij,
//		double I) {
//
//	double3 force = make_double3(0.0);
//	double3 partToPartForce;
//	double3 partToPartTorque = make_double3(0.0);
//
//	uint pair;
//	uint j = numParticles - 1;
//	if (indice_i != j) {
//		double3 pos2 = make_double3(FETCH(dPos, j));
//		double3 vel2 = make_double3(FETCH(dVel, j));
//		double4 posA2 = FETCH(dPosA, j);
//		double3 velA2 = make_double3(FETCH(dVelA, j));
//		uint vecino = FETCH(dPos, j).w;
//
//		int indice_vecino;
//		uint row = indice_i*NUMAX_CONT;
//
//		uint found=0;
//		uint p=0;
//
//		while(p<NUM_VEC[indice_i] && found==0) {
//			if (m_TANG_indice[row + p]==vecino) {
//				// indice_vecino = p;
//				// pair=(row + indice_vecino);
//				pair = row + p;
//				found = 1;
//			}
//			p++;
//		}
//		if(found==0) {
//			// indice_vecino = NUM_VEC[indice_i];
//			// pair=(row + indice_vecino);
//			pair = row + NUM_VEC[indice_i];
//			m_TANG_indice[pair] = vecino;
//			m_dTangSpring[pair] = make_double3(0.0);
//			NUM_VEC[indice_i]++;
//		}
//
//		double3 forceAppPoint = pointOfForceApplication(pos, pos2);
//
//		// collide two spheres
//		double R1 = FETCH(dVel, indice_i).w;
//		double R2 = FETCH(dVel, j).w;
//
//		partToPartForce = collideSpheres(pos, pos2, posA, posA2,vel, vel2, velA,velA2, R1, R2, m_dTangSpring, pair, numParticles*numParticles);
//                 force += partToPartForce;
//
//                 m_dForce_ij[pair].x = partToPartForce.x;
//                 m_dForce_ij[pair].y = partToPartForce.y;
//                 m_dForce_ij[pair].z = partToPartForce.z;
//                 m_dForce_ij[pair].w = (double)indice_i;
//
//                 m_dContact_ij[pair].x = forceAppPoint.x;
//                 m_dContact_ij[pair].y = forceAppPoint.y;
//                 m_dContact_ij[pair].z = forceAppPoint.z;
//                 m_dContact_ij[pair].w = (double)vecino;
//
//
//                 partToPartTorque = cross(forceAppPoint - pos,partToPartForce);
//                 double4 PosAtemp= make_double4(posA.x,-posA.y,-posA.z,-posA.w);
//                 partToPartTorque = rotate(PosAtemp,partToPartTorque);
//
//                 torque[indice_i].x += partToPartTorque.x/I ;
//                 torque[indice_i].y += partToPartTorque.y/I ;
//                 torque[indice_i].z += partToPartTorque.z/I ;
//                // torque[indice_i].w = 0.0 ;
//
//                 if (m_dTangSpring[pair].x==0.0 && m_dTangSpring[pair].y==0.0 && m_dTangSpring[pair].z==0.0){
//                   NUM_VEC[indice_i]--;
//                   if(NUM_VEC[indice_i]!=0) {
//                         uint pair_to_move =   row + NUM_VEC[indice_i];
//                         m_TANG_indice[pair] = m_TANG_indice[pair_to_move];
//                         m_dTangSpring[pair] = m_dTangSpring[pair_to_move];
//                   }
//                 }
//
//
//            }
//
//	return force;
//}

__device__
double3 sixs_flat_walls(double3 pos, int wall, double3 NORMAL, double3 pp) {

	double3 pos2 = pp;
	double3 NORMAL_VECTOR = NORMAL;
	double3 vertex_to_particle;
	vertex_to_particle = (pos - pos2);

	// pointing the point
	double where = dot(vertex_to_particle, NORMAL_VECTOR); // for answering where is the plane_pointing
	if (where > 0) { // they are pointing to the same, that's wrong
		NORMAL_VECTOR = -NORMAL_VECTOR;
	}

	double D = -1.0 * dot(NORMAL_VECTOR, pos2);
	double A = NORMAL_VECTOR.x;
	double B = NORMAL_VECTOR.y;
	double C = NORMAL_VECTOR.z; ///  I have the plane

	double DISTANCE = A * pos.x + B * pos.y + C * pos.z + D; /// distance to the plain
	DISTANCE = fabs(DISTANCE);

	pos2 = pos + DISTANCE * NORMAL_VECTOR; // That is closest point

	return pos2;
}

__device__
double3 wall_cylinder(double3 pos) {
	double3 pos2;
	double DIST2 = pos.x * pos.x + pos.z * pos.z;
	// printf("%f %f \n",DIST2,CYLINDER_RADI*CYLINDER_RADI);
	double CYLINDER_RADIUS_DIST = CYLINDER_RADIUS / sqrt(DIST2);
	double cos_phi = pos.x * CYLINDER_RADIUS_DIST;
	double sin_phi = pos.z * CYLINDER_RADIUS_DIST;
	pos2 = make_double3(cos_phi, pos.y, sin_phi);
	return pos2;
}

//__device__
//double3 collide_wall(double3 pos, double3 vel, double4 posA, double3 velA,
//		double4* torque, int wall, double3 NV, double3 pp, int indice_i,
//		double3* m_dTangSpring_W, uint N, double4* m_dForce_W,
//		double4* m_dContact_W, double II) {
//	double3 force = make_double3(0.0);
//	double3 partToPartForce;
//	double3 partToPartTorque = make_double3(0.0);
//
//	// double3
//	double3 pos2;
//	if (wall < 6)
//		pos2 = sixs_flat_walls(pos, wall, NV, pp);
//	else
//		pos2 = wall_cylinder(pos);
//
//	double3 vel2 = make_double3(0.0); // make_double3(FETCH(oldVel, j));
//	double3 velA2 = make_double3(0.0); // make_double3(FETCH(oldVelA, j));
//	double4 posA22 = make_double4(0.0, 1.0, 0.0, 0.0); // FETCH(oldPosA, j);
//	int pair = N * wall + indice_i;
//
//	partToPartForce = collideSpheres_wall(pos, pos2, posA, posA22, vel, vel2,
//			velA, velA2, params.particleRadius, params.particleRadius,
//			m_dTangSpring_W, pair);
//
//	force += partToPartForce;
//	double3 forceAppPoint = pointOfForceApplication(pos, pos2);
//
//	m_dForce_W[pair].x = partToPartForce.x;
//	m_dForce_W[pair].y = partToPartForce.y;
//	m_dForce_W[pair].z = partToPartForce.z;
//	m_dForce_W[pair].w = (double) indice_i;
//
//	m_dContact_W[pair].x = forceAppPoint.x;
//	m_dContact_W[pair].y = forceAppPoint.y;
//	m_dContact_W[pair].z = forceAppPoint.z;
//	m_dContact_W[pair].w = (double) wall;
//
//	partToPartTorque = cross(forceAppPoint - pos, partToPartForce);
//	double4 PosAtemp = make_double4(posA.x, -posA.y, -posA.z, -posA.w);
//	partToPartTorque = rotate(PosAtemp, partToPartTorque);
//
//	torque[indice_i].x += partToPartTorque.x / II;
//	torque[indice_i].y += partToPartTorque.y / II;
//	torque[indice_i].z += partToPartTorque.z / II;
//	// torque[indice_i].w = 0.0 ;
//
//	return force;
//}

//__global__
//void collideD(double4* Acc, double4* AccA,
//		double4* oldPos, // input: sorted positions
//		double4* oldPosA, // input: sorted positions
//		double4* oldVel, // input: sorted velocities
//		double4* oldVelA, // input: sorted velocities
//		double4* oldAcc, // input: sorted accelerations
//		double4* oldAccA, // input: sorted accelerations
//		uint* gridParticleIndex, // input: sorted particle indices
//		uint* cellStart, uint* cellEnd, uint numParticles,
//		double3* m_dTangSpring, int* NUM_VEC, int* m_TANG_indice,
//		double4* m_dForce_ij, double4* m_dContact_ij) {
//	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (index >= numParticles)
//		return;
//
//	// read particle data from sorted arrays
//
//	double3 pos = make_double3(FETCH(oldPos, index));
//	double4 posA = FETCH(oldPosA, index);
//	double3 vel = make_double3(FETCH(oldVel, index));
//	double3 velA = make_double3(FETCH(oldVelA, index));
//
//	// get address in grid
//	int3 gridPos = calcGridPos(pos);
//
//	// examine neighbouring cells
//	double3 force = make_double3(0.0);
//
//	uint originalIndex = gridParticleIndex[index];
//
//	AccA[originalIndex].x = 0.0;
//	AccA[originalIndex].y = 0.0;
//	AccA[originalIndex].z = 0.0;
//	// AccA[originalIndex].w = 0.0 ;
//
//	int LIM = (int) AccA[originalIndex].w;
//
//	for (int z = -LIM; z <= LIM; z++) {
//		for (int y = -LIM; y <= LIM; y++) {
//			for (int x = -LIM; x <= LIM; x++) {
//
//				int3 neighbourPos = gridPos + make_int3(x, y, z);
//
//				force += collideCell(neighbourPos, index, originalIndex, pos,
//						vel, posA, velA, oldPos, oldVel, oldPosA, oldVelA,
//						cellStart, cellEnd, AccA, m_dTangSpring, numParticles,
//						NUM_VEC, m_TANG_indice, m_dForce_ij, m_dContact_ij);
//			}
//		}
//	}
//
//	double masa = FETCH(oldVelA, index).w;
//	double radio = FETCH(oldVel, index).w;
//	double I = (2. / 5.) * masa * radio * radio;
//
//	uint originalIndex1 = gridParticleIndex[index];
//
//	Acc[originalIndex1].x = force.x / masa; // esta es la masa  Vel[originalIndex].w
//	Acc[originalIndex1].y = force.y / masa + GRAVITY;
//	Acc[originalIndex1].z = force.z / masa;
//	//  Acc[originalIndex].w =  0.0;
//
//	AccA[originalIndex1].x = AccA[originalIndex1].x / I;
//	AccA[originalIndex1].y = AccA[originalIndex1].y / I;
//	AccA[originalIndex1].z = AccA[originalIndex1].z / I;
//}

////collide with the container limits
//__global__
//void collideD1(double4* Acc, double4* AccA,
//		double4* oldPos, // input: sorted positions
//		double4* oldPosA, // input: sorted positions
//		double4* oldVel, // input: sorted velocities
//		double4* oldVelA, // input: sorted velocities
//		double4* oldAcc, // input: sorted accelerations
//		double4* oldAccA, // input: sorted accelerations
//		uint* gridParticleIndex, // input: sorted particle indices
//		uint* cellStart, uint* cellEnd, uint numParticles,
//		double3* m_dTangSpring_W, double4* m_dForce_W, double4* m_dContact_W) {
//	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (index >= numParticles)
//		return;
//	double3 force = make_double3(0.0);
//
//	uint originalIndex = gridParticleIndex[index];
//	// read particle data from sorted arrays
//
//	double3 pos = make_double3(FETCH(oldPos, index));
//	double4 posA = FETCH(oldPosA, index);
//	double3 vel = make_double3(FETCH(oldVel, index));
//	double3 velA = make_double3(FETCH(oldVelA, index));
//	double3 pos2;
//	double3 NORMAL;
//
//	double masa = FETCH(oldVelA, index).w;
//	double radio = FETCH(oldVel, index).w;
//	double I = (2. / 5.) * masa * radio * radio;
//
//	//collide with the bottom of the cube
//
//	if (pos.y < params.ymin) {
//		pos2 = make_double3(0.0, -SYSTEMSIZE - PARTICLE_RADIUS, 0.0);
//		NORMAL = make_double3(0.0, 1.0, 0.0);
//		force += collide_wall(pos, vel, posA, velA, AccA, 2, NORMAL, pos2,
//				originalIndex, m_dTangSpring_W, numParticles, m_dForce_W,
//				m_dContact_W, I);
//	}
//	/* if(pos.y>params.ymax ) {
//	 pos2   = make_double3(0.0, SYSTEM_SIZE+PARTICLE_RADIUS,0.0);
//	 NORMAL =  make_double3(0.0,1.0,0.0);
//	 force += collide_wall(pos,vel, posA,velA,AccA,3,NORMAL,pos2,originalIndex,m_dTangSpring_W,numParticles,m_dForce_W,m_dContact_W,I);
//	 }*/
//
//	//collide with thje cylinder walls
//	double where2 = pos.x * pos.x + pos.z * pos.z;
//	if (where2 > CILINDER_RADIDelta2) {
//		force += collide_wall(pos, vel, posA, velA, AccA, 6, NORMAL, pos2,
//				originalIndex, m_dTangSpring_W, numParticles, m_dForce_W,
//				m_dContact_W, I);
//	}
//
//	Acc[originalIndex].x += force.x / masa;
//	Acc[originalIndex].y += force.y / masa;
//	Acc[originalIndex].z += force.z / masa;
//	// Acc[originalIndex].w =  0.0;
//
//}

//__global__
//void collideD2(
//		double4* dPos, // input: sorted positions
//		double4* dPosA, // input: sorted positions
//		double4* dVel, // input: sorted velocities
//		double4* dVelA, // input: sorted velocities
//		double4* dAcc, // input: sorted accelerations
//		double4* dAccA, // input: sorted accelerations
//		uint* gridParticleIndex, // input: sorted particle indices
//		uint* cellStart, uint* cellEnd, uint numParticles,
//		double3* m_dTangSpring, int* NUM_VEC, int* m_TANG_indice,
//		double4* m_dForce_ij, double4* m_dContact_ij) {
//	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (index >= numParticles)
//		return;
//	double3 force = make_double3(0.0);
//
//	uint originalIndex = gridParticleIndex[index];
//	// read particle data from sorted arrays
//
//	double3 pos = make_double3(FETCH(dPos, originalIndex));
//	double4 posA = FETCH(dPosA, index);
//	double3 vel = make_double3(FETCH(dVel, originalIndex));
//	double3 velA = make_double3(FETCH(dVelA, originalIndex));
//	double3 pos2;
//	double3 NORMAL;
//
//	double masa = FETCH(dVelA, originalIndex).w;
//	double radio = FETCH(dVel, originalIndex).w;
//	double I = (2. / 5.) * masa * radio * radio;
//
//	force = collide_BOLA(originalIndex, pos, vel, posA, velA, dPos, dVel, dPosA,
//			dVelA, cellStart, cellEnd, dAccA, m_dTangSpring, numParticles,
//			NUM_VEC, m_TANG_indice, m_dForce_ij, m_dContact_ij, I);
//
//	dAcc[originalIndex].x += force.x / masa;
//	dAcc[originalIndex].y += force.y / masa;
//	dAcc[originalIndex].z += force.z / masa;
//	//  dAcc[originalIndex].w =  0.0;
//
//}

#endif

