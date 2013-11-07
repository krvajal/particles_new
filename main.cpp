
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <iterator>
#include "ParticleSystem.h"
#include <cutil_math.h>
#include "config_system.h"
#include "Sphere.h"

const int NUM_PARTICLES=1000;



using namespace std;

int main(int argc, char*argv[]){




  ifstream input_stream("init.txt");

  double3 G;
  double timestep;
  double time;
  uint nstep;
  uint nprint;
  uint nenergy;
  double lx,ly,lz,x0,y0,z0;


  while (input_stream.peek() == '#')
    {
      string type;
      input_stream >> type;
      if (type == "#gravity:")
      {
        input_stream >> G.x>> G.y >> G.z;
        input_stream.ignore(100, '\n');
        cout << "gravity: " << G << endl;
      }
      else if (type == "#time:")
      {
        input_stream >> time;
        input_stream.ignore(100, '\n');
        cout << "time: " << time << endl;
      }
      else if (type == "#nstep:")
      {
        input_stream >> nstep;
        input_stream.ignore(100, '\n');
        cout << "nstep: " << nstep << endl;
      }
      else if (type == "#timestep:")
      {
        input_stream >> timestep;
        input_stream.ignore(100, '\n');
        cout << "timestep: " << timestep << endl;
      }
      else if (type == "#nprint:")
      {
        input_stream >> nprint;
        input_stream.ignore(100, '\n');
        cout << "nprint: " << nprint << endl;
      }
      else if (type == "#nenergy:")
      {
        input_stream >> nenergy;
        input_stream.ignore(100, '\n');
        cout << "nenergy: " << nenergy << endl;
      }
      else if (type == "#lx:")
      {
        input_stream >> lx;
        input_stream.ignore(100, '\n');
        cout << "lx: " << lx << endl;
      }
      else if (type == "#ly:")
      {
        input_stream >> ly;
        input_stream.ignore(100, '\n');
        cout << "ly: " << ly << endl;
      }
      else if (type == "#x0:")
      {
        input_stream >> x0;
        input_stream.ignore(100, '\n');
        cout << "x0: " << x0 << endl;
      }
      else if (type == "#y0:")
      {
        input_stream >> y0;
        input_stream.ignore(100, '\n');
        cout << "y_0: " << y0 << endl;
      }
      else
      {
        cerr << "init: unknown global property: " << type << endl;
        abort();
      }
    }
  vector<Sphere> particles;
  copy(istream_iterator<Sphere>(input_stream),istream_iterator<Sphere>(),back_inserter(particles));

  input_stream.close();

  uint num_particles=particles.size();

  double3 system_size=make_double3(lx,ly,lz);
  double3 system_origin=make_double3(x0,y0,z0);


  ParticleSystem *particle_system=new ParticleSystem(num_particles,make_uint3(10,10,10),system_size,system_origin,timestep,false);

  double * pos= new double(num_particles*4);
  double * vel= new double(num_particles*4);
  double * acc= new double(num_particles*4);
  double * r  = new double(num_particles);
  double *mass= new double(num_particles);

  memset(pos,0,sizeof(double)*num_particles*4);
  memset(vel,0,sizeof(double)*num_particles*4);
  memset(acc,0,sizeof(double)*num_particles*4);

  memset(r,0,sizeof(double)*num_particles);
  memset(mass,0,sizeof(double)*num_particles);

  for(int i=0;i<num_particles;i++){

	  pos[4*i+0]=particles[i].pos().x;
	  pos[4*i+1]=particles[i].pos().y;
	  pos[4*i+2]=particles[i].pos().z;


  }
  particle_system->setArray(ParticleSystem::ParticleArray::POSITION,pos,0,num_particles);
  particle_system->setArray(ParticleSystem::ParticleArray::VELOCITY,vel,0,num_particles);
  particle_system->setArray(ParticleSystem::ParticleArray::ACCELERATION,acc,0,num_particles);
  particle_system->setArray(ParticleSystem::ParticleArray::RADIUS,r,0,num_particles);
  particle_system->setArray(ParticleSystem::ParticleArray::MASS,mass,0,num_particles);

  ofstream fout("phase.dat");

  for(int i=0;i<nstep;i++){

       particle_system->update(timestep);
       if(i%nprint==0){

    	   fout << "#NewFrame\n";
    	   fout << "#no_of_particles: " << num_particles << endl;
    	   fout << "#compressed: no\n";
    	   fout << "#type: SphereXYPhiVxVyOmegaRMFixed25\n";
    	   fout << "#gravity: " << G.x << " " << G.y << " " << G.z << endl;
    	   fout << "#time: " << i*timestep << endl;
    	   fout << "#timestep: " << timestep << endl;
    	   fout << "#EndOfHeader\n";
    	   ;

    	   particle_system->dumpParticlesToFile(fout,0,num_particles);

       }
  }

  fout.close();

  delete [] pos;
  delete [] mass;
  delete [] vel;
  delete [] acc;
  delete [] r;
}
