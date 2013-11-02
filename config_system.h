#ifndef CONFIG_SYSTEM_H
#define CONFIG_SYSTEM_H


//the number of contacts each particle can have
#define NUMAX_CONT 64

#define COLLIDE_SPRING 1.0
#define TIMESTEP 1e-6

#define MAX(a,b) ((a)>(b))?(a):(b)
#define SYSTEMSIZEX 1.0
#define SYSTEMSIZEY 1.0
#define SYSTEMSIZEZ 1.0
#define SYSTEMSIZE MAX(SYSTEMSIZEX,MAX(SYSTEMSIZEY,SYSTEMSIZEZ))

#define MU 0.1
#define MU_WALL 0.1
#define GRAVITY 9.81
#define CYLINDER_RADIUS .5
#define PARTICLE_RADIUS 0.03



#ifndef __CONFIG_SYSTEM_GEOMETRY_H__
#define __CONFIG_SYSTEM_GEOMETRY_H__



#ifndef USE_PBC
    #define NOUSE_PBC
#endif

#ifndef NUMAX_CONT
    #define NUMAX_CONT        50  // maxim number of contacts
#endif


#ifndef PARTICLE_RADIUS
    #define PARTICLE_RADIUS   (0.005) // 1.0/64.0
#endif



#ifndef GRID_SIZE
    #define GRID_SIZE        SYSTEM_SIZE/PARTICLE_RADIUS
#endif



#ifndef R_BOLA
     #define R_BOLA     0.023
#endif

#ifndef CYLINDER_RADIUS
   #define CYLINDER_RADIUS         .12
#endif

#ifndef CYLINDER_RADIDelta2
   #define CILINDER_RADIDelta2   (CYLINDER_RADIUS-2.*PARTICLE_RADIUS)*(CYLINDER_RADIUS-2.*PARTICLE_RADIUS)
#endif



#ifndef MIND
    #define MIND     (2*PARTICLE_RADIUS)          // 2*PARTICLE_RADIUS
#endif

#ifndef MIND2
    #define MIND2    (MIND*MIND)   // MIN²
#endif

#ifndef MB
    #define MB      (R_BOLA+PARTICLE_RADIUS)// MIN²
#endif


#ifndef MB2
    #define MB2       (R_BOLA+PARTICLE_RADIUS)*(R_BOLA+PARTICLE_RADIUS) // MIN²
#endif


#endif //__CONFIG_SYSTEM_GEOMETRY_H__


#ifndef __CONFIG_GAS_H__
#define __CONFIG_GAS_H__

#ifndef NUM_FREE
    #define NUM_FREE           1000000    // simulation without dammping
#endif

// MAXIMUM NUM_PARTICLES = GRID_SIZE*GRID_SIZE*GRID_SIZE
                                     // TODAY MAXIMUN = 64*64*64 = 262144
#ifndef NUM_PARTICLES
    #define NUM_PARTICLES      10000
#endif

#ifndef NUM_ITERATIONS
    #define NUM_ITERATIONS     10000000
#endif

#ifndef SHUT_TIME
   #define  SHUT_TIME         500000
#endif


#ifndef TIMESTEP
    #define TIMESTEP           1.0e-6
#endif

#ifndef COLLISION_TIME
#define COLLISION_TIME 1e-5
#endif

#ifndef GRAVITY
    #define GRAVITY             -9.81
#endif


#ifndef COE_REST
    #define COE_REST_N           1.0
#endif

#ifndef COE_REST
    #define COE_REST_T           0.2
#endif


#ifndef YOUNG
    #define YOUNG               1.0e7
#endif


#ifndef COLLIDE_SPRING
    #define COLLIDE_SPRING       ((4./3.)*YOUNG*sqrt(PARTICLE_RADIUS/2.))
#endif

/// Este es el cambio

#ifndef BETA
    #define BETA                (2.*sqrt(5.0/6.0)*1./sqrt(1.0+pow(3.14/log(COE_REST_N),2.0)))
#endif

#ifndef Sn
    #define Sn                  (2.*YOUNG*sqrt(PARTICLE_RADIUS/2.0))
#endif

///

#ifndef DENSITY
    #define DENSITY             14.0
#endif




#ifndef DENSITY2
 #define DENSITY2        (DENSITY*50)
#endif

#ifndef MASS2
    #define MASS2        ((4./3.)*3.1415*DENSITY2*R_BOLA*R_BOLA*R_BOLA)
#endif

#ifndef MI
    #define MI            2.18e-8 // ((2./5.)*MASS*PARTICLE_RADIUS*PARTICLE_RADIUS)
#endif

///  Esto es para Hooke
#ifndef DAMPING_N
     #define DAMPING_N         pow(2.*COLLIDE_SPRING*MASS/(pow(3.1415/log(1./COE_REST_N),2.0)+1.0),0.5)
#endif

#ifndef MU
    #define MU                   0.5
#endif

#ifndef MU2
    #define MU2                  0.25
#endif

#ifndef MU_WALL
    #define MU_WALL              0.5
#endif

#ifndef MU_WALL2
    #define MU_WALL2             0.25     // MU*MU
#endif

#ifndef TEMPERATURE_R
    #define TEMPERATURE_R        10.0
#endif

#ifndef TEMPERATURE_T
    #define TEMPERATURE_T        10.0
#endif

#ifndef GAMMA_N
#define GAMMA_N -2.0*log(COE_REST_N)/COLLISION_TIME
#endif


#define sqr(x) ((x)*(x))

#ifndef K_N
#define K_N (sqr(M_PI)+sqr(log(COE_REST_N))/sqr(COLLISION_TIME))
#endif

#ifndef DELTA_TIME
#define DELTA_TIME 1e-7
#endif


#define ROTATION

#endif //__CONFIG_GAS_H__

#endif
