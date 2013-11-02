#ifndef SIM_PARAMS_H
#define SIM_PARAMS_H



#ifndef __DEVICE_EMULATION__
#define USE_TEX 0
#endif

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#include "vector_types.h"


typedef unsigned int uint;

// simulation parameters
struct SimParams {

    double3 gravity;

    double particleRadius;
    double mass;
    double momentofinertia;

    uint3 gridSize;
    uint numCells;
    double3 worldOrigin;
    double3 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

    double spring;
    double NormalDamping;
    double shear;
    double Tangentdamping;

    double attraction;
    double boundaryDamping;
    double GlobalDamping;

    double xmax;
    double xmin;

    double ymax;
    double ymin;

    double zmax;
    double zmin;

    double LX;
    double LY;
    double LZ;

    double X0;
    double Y0;
    double Z0;

    double L;

    double pLX;
    double pLY;
    double pLZ;

    double mu;
    double muw;

    double hole;
    double R_hole;
    double R_hole2;

    int nmaxc;

    double dt;
    double dt_2;

    double MIN_D;
    double MIN_D2;



};




#endif
