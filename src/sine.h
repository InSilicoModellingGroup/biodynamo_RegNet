#ifndef SINE_H_
#define SINE_H_

#include <iostream>
#include <boost/array.hpp>

#include "boost/numeric/odeint.hpp"
#include "boost/phoenix/core.hpp"
#include "boost/phoenix/operator.hpp"

typedef boost::numeric::ublas::vector<double>  boost_vector_t;
typedef boost::numeric::ublas::matrix<double>  boost_matrix_t;

namespace sine {

struct ODE_system {
  void operator()(const boost_vector_t& x, boost_vector_t& dxdt, double t) const {
    dxdt[0] = A*cos(t);
  }
  //
  const double A = 10.0;
};

struct ODE_output {
  void operator()(const boost_vector_t& x, double t) {
    std::clog << t << ',' << x[0] << std::endl;
  }
};

inline int Simulate(int argc, const char** argv) {
  boost_vector_t x(1);
  x[0] = 0.0;

  typedef boost::numeric::odeint::runge_kutta_dopri5<boost_vector_t> ode_int;

  // set-up the Runge-Kutta integrator
  auto stepper =
    boost::numeric::odeint::make_dense_output<ode_int>(1.e-6,1.e-6);

  // perform the time-integration
  integrate_const(
    stepper, ODE_system(), x,
    0.0, 12.5663706144, 0.001,
    ODE_output()
  );

  std::cout << "Simulation completed successfully!\n" << std::endl;
  return 0;
}

} // namespace sine

#endif // SINE_H_
