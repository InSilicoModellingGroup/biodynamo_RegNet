#ifndef OSCILLATOR_H_
#define OSCILLATOR_H_

#include <iostream>
#include <boost/array.hpp>

#include "boost/numeric/odeint.hpp"
#include "boost/phoenix/core.hpp"
#include "boost/phoenix/operator.hpp"

typedef boost::numeric::ublas::vector<double>  boost_vector_t;
typedef boost::numeric::ublas::matrix<double>  boost_matrix_t;

namespace oscillator {

// https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
double mu = 100.0;

struct ODE_system {
  void operator()(const boost_vector_t& x, boost_vector_t& dxdt, double t) const {
    dxdt[0] = x[1];
    dxdt[1] = mu * x[1] - mu * x[0] * x[0] * x[1] - x[0];
  }
};

struct ODE_jacobian {
  void operator()(const boost_vector_t& x, boost_matrix_t& jac, double t, boost_vector_t& dfdt) const {
    jac(0, 0) = 0.0;
    jac(0, 1) = 1.0;
    jac(1, 0) = -2.0 * mu * x[0] * x[1] - 1.0;
    jac(1, 1) = mu - mu * x[0] * x[0];
    //
    dfdt[0] = dfdt[1] = 0.0;
  }
};

struct ODE_output {
  void operator()(const boost_vector_t& x, double t) {
    std::clog << t << ',' << x[0] << ',' << x[1] << std::endl;
  }
};

inline int Simulate(int argc, const char** argv) {
  boost_vector_t xy(2);
  xy[0] = 2;
  xy[1] = 0;

  typedef boost::numeric::odeint::rosenbrock4<double> ode_int;

  // set-up the Rosenbrock integrator
  auto stepper =
    boost::numeric::odeint::make_dense_output<ode_int>(1.e-6,1.e-6);

  // perform the time-integration
  integrate_const(
    stepper, std::make_pair(ODE_system(), ODE_jacobian()), xy,
    0.0, 500.0, 1.0e-2,
    ODE_output()
  );

  std::cout << "Simulation completed successfully!\n" << std::endl;
  return 0;
}

} // namespace oscillator

#endif // OSCILLATOR_H_
