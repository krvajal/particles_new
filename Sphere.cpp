
#include <cmath>
#include "Sphere.h"

#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include "Vector.h"

extern double3 G;

using namespace std;
istream & operator >>(istream & is, Sphere & p)
{
	is >> p.m_pos.x >>p.m_pos.y>>p.m_pos.z;
	is >> p.m_vel.x >>p.m_vel.y>>p.m_vel.z;

	is.ignore(2048,'\n');


	//
	//  p.rtd2 = null;
	//  p.rtd3 = null;
	//  p.rtd4 = null;
	//  p._force = null;

	//compute inertia moment


	return is;
}

ostream & operator <<(ostream & os, const Sphere & p)
{
	os << p.m_pos.x <<'\t'<<p.m_pos.y<<'\t'<<p.m_pos.z<<'\t';
	os<< p.m_vel.x <<'\t'<<p.m_vel.y<<'\t'<<p.m_vel.z<<'\t';
	os<< p.m_acc.x <<'\t'<<p.m_acc.y<<'\t'<<p.m_acc.z<<'\t';
	os << p.m_r << '\t' << p.m_mass << " " << p.m_ptype <<endl;

	return os;
}
