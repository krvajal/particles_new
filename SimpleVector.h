

#ifndef _Vector_h
#define _Vector_h

#include <math.h>
#include <iostream>

using namespace std;

class double3 {
  friend istream & operator >> (istream & is, double3 & v) {
    is >> v.x >> v.y >> v.phi;
    return is;
  }
  friend ostream & operator << (ostream & os, const double3 & v) {
    os << v.x << " " << v.y << " " << v.phi;
    return os;
  }
  friend double3 operator + (const double3 & v1, const double3 & v2) {
    double3 res(v1);
    res+=v2;
    return res;
  }
  friend double3 operator - (const double3 & v1, const double3 & v2) {
    double3 res(v1);
    res-=v2;
    return res;
  }
  friend double3 operator * (double c, const double3 & p) {
    double3 res=p;
    res*=c;
    return res;
  }
  friend double3 operator * (const double3 & p, double c) {
    return c*p;
  }
  friend double norm2d(const double3 & v) {
    return sqrt(v.x*v.x+v.y*v.y);
  }
  friend double dot(const double3 & v1, const double3 & v2) {
    return v1.x*v2.x + v1.y*v2.y;
  }
  friend double cross(const double3 & v1, const double3 & v2) {
    return v1.x*v2.y-v1.y*v2.x;
  }

public:
  explicit double3(double x=0,double y=0,double phi=0): x(x), y(y), phi(phi){};



  const double3 & operator += (const double3 & p){
    x+=p.x; y+=p.y; phi+=p.phi;
    return *this;
  }
  const double3 & operator -= (const double3 & p){
    x-=p.x; y-=p.y; phi-=p.phi;
    return *this;
  }
  const double3 & operator *= (double c){
    x*=c; y*=c; phi*=c;
    return *this;
  }


public:
  //the data members

  double x,y,phi;
};

const double3 null(0,0,0);
#endif
