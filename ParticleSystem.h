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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include "SimParams.h"
#include "vector_functions.h"
#include <cstdio>
#include <fstream>

// Particle system class
class ParticleSystem
{
public:
	ParticleSystem(uint numParticles, uint3 gridSize, double3 system_size, double3 system_origin, double timestep,bool bUseOpenGL);
	~ParticleSystem();

	enum ParticleConfig {CONFIG_RANDOM, CONFIG_FROM_FILE, CONFIG_GRID, _NUM_CONFIGS	};

	enum ParticleArray {POSITION, VELOCITY, ACCELERATION, POSITIONA, VELOCITYA, ACCELERATIONA,RADIUS,MASS};

	void update(double deltaTime);
	void reset(ParticleConfig config);

	double* getArray(ParticleArray array);

	void setArray(ParticleArray array, const double* data, int start,
			int count);

	int getNumParticles() const {
		return m_numParticles;
	}

	void dumpGrid();

	void dumpParticles(uint start, uint count);

	void dumpParticlesToFile(std::ofstream &os,uint start, uint count);
//	void golpe();
	void save_coolling();
	void save_data();
	void coolling();

	void setIterations(int i) {
		m_solverIterations = i;
	}

	void setNormalDamping(double x) {
		m_params.NormalDamping = x;
	}
	void setGravity(double x) {
		m_params.gravity = make_double3(0.0, x, 0.0);
	}

	void setCollideSpring(double x) {
		m_params.spring = x;
	}
	void setTangentDamping(double x) {
		m_params.Tangentdamping = x;
	}
	void setCollideShear(double x) {
		m_params.shear = x;
	}
	void setCollideAttraction(double x) {
		m_params.attraction = x;
	}
	void setGlobalDamping(double x) {
		m_params.GlobalDamping = x;
	}

	void updateCardParams();


	double getParticleRadius() {
		return m_params.particleRadius;
	}

	uint3 getGridSize() {
		return m_params.gridSize;
	}
	double3 getWorldOrigin() {
		return m_params.worldOrigin;
	}
	double3 getCellSize() {
		return m_params.cellSize;
	}

	//*******************************************************

	void dump_phase(FILE *f);

	//*******************************************************
protected:
	// methods
	ParticleSystem() {
	}

	void _initialize(int numParticles);
	void _finalize();

	void initGrid(uint *size, double spacing, double jitter, uint numParticles);

protected:
	// data
	bool m_bInitialized, m_bUseOpenGL;
	uint m_numParticles;

	// CPU data
	double* m_hPos; // particle positions
	double* m_hVel; // particle velocities
	double* m_hAcc; // particle accelerations

	double* m_hPosA; // particle positions
	double* m_hVelA; // particle velocities
	double* m_hAccA; // particle accelerations

	double* m_hParticleRadius;
	double* m_hParticleMass;
	double* m_hFxy; // particle force
	double* m_hContact; // particle point of application

	double* m_hFwall; // particle force
	double* m_hContact_W; // particle point of application

	uint* m_hParticleHash;
	uint* m_hCellStart;
	uint* m_hCellEnd;






	// GPU data
	double* m_dPos;
	double* m_dVel;
	//  double* m_dVelINT;
	double* m_dAcc;

	double* m_dPosA;
	double* m_dVelA;
	//  double* m_dVelAINT;
	double* m_dAccA;

	double* m_dSortedPos;
	double* m_dSortedVel;
	double* m_dSortedAcc;

	double* m_dSortedPosA;
	double* m_dSortedVelA;
	double* m_dSortedAccA;

	double* m_dTangSpring;
	double* m_dTangSpring_W;
	int* NUM_VEC;
	int* m_TANG_indice;

	double* m_dForce_ij;
	double* m_dCONTACT_ij;

	double* m_dForce_W;
	double* m_dCONTACT_W;
	double* m_dParticleRadius;
	uint m_iterations;
	uint m_updFrequency;

	// grid data for sorting method
	uint* m_dGridParticleHash; // grid hash value for each particle
	uint* m_dGridParticleIndex; // particle index for each particle
	uint* m_dCellStart; // index of start of each cell in sorted list
	uint* m_dCellEnd; // index of end of cell
	double *m_dMass;
	double *m_dR;
	uint m_gridSortBits;

	// params
	SimParams m_params;
	uint3 m_gridSize;
	uint m_numGridCells;

	uint m_timer;

	uint m_solverIterations;
};

#endif // __PARTICLESYSTEM_H__
