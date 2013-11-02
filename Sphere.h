

#ifndef _Sphere_h
#define _Sphere_h

#include <iostream>
#include <stdexcept>

#include "Vector.h"




class Sphere {

	friend std::istream & operator >> (std::istream & is, Sphere & p);
	friend std::ostream & operator << (std::ostream & os, const Sphere & p);


public:


	double3 & pos() {return m_pos;}
	double3 pos() const {return m_pos;)
	double3  vel() const {return m_vel;}
	double3 & velA() {return m_velA;}

	void set_radius(double new_radius)
	{
		if(new_radius<=0.0)
			throw std::invalid_argument("the radius can not be negative or zero");
		m_r=new_radius;
	}
	double r() const {return m_r;}

	double mass() const {return m_mass;}

	void set_mass(double mass)
	{
		if(mass<=0.0)
			throw std::invalid_argument("the mass can not be negative");
		m_mass=mass;

	}
	int ptype() const {return m_ptype;}


//public:
//	//properties of the particle material
//	double Y,A,mu,gamma;

private:

	double m_r, m_mass;
	int m_ptype;

	double3 m_pos,m_vel,m_acc;
	double3 m_posA,m_velA,m_accA;

};

#endif
