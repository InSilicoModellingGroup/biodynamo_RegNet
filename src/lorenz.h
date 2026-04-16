#ifndef LORENZ_H_
#define LORENZ_H_

#include <iostream>
#include <boost/array.hpp>

#include "boost/numeric/odeint.hpp"
#include "boost/phoenix/core.hpp"
#include "boost/phoenix/operator.hpp"

// https://en.wikipedia.org/wiki/Lorenz_system
const double sigma = 10.0;
const double rho = 28.0;
const double beta = 8.0 / 3.0;

typedef boost::numeric::ublas::vector<double>  boost_vector_t;
typedef boost::numeric::ublas::matrix<double>  boost_matrix_t;

namespace lorenz {

struct ODE_system {
  void operator()(const boost_vector_t& x, boost_vector_t& dxdt, double t) const {
    dxdt[0] = sigma * x[1] - sigma * x[0];
    dxdt[1] = rho * x[0] - x[1] - x[0] * x[2];
    dxdt[2] = -beta * x[2] + x[0] * x[1];
  }
};

struct ODE_jacobian {
  void operator()(const boost_vector_t& x, boost_matrix_t& jac, double t, boost_vector_t& dfdt) const {
    jac(0, 0) = -sigma;
    jac(0, 1) = sigma;
    jac(0, 2) = 0.0;
    jac(1, 0) = rho  - x[2];
    jac(1, 1) = -1.0;
    jac(1, 2) = -x[0];
    jac(2, 0) = x[1];
    jac(2, 1) = x[0];
    jac(2, 2) = -beta;
    //
    dfdt[0] = 0.0;
    dfdt[1] = 0.0;
    dfdt[2] = 0.0;
  }
};

struct ODE_output {
  void operator()(const boost_vector_t& x, double t) {
    std::clog << t << ',' << x[0] << ',' << x[1] << ',' << x[2] << std::endl;
  }
};

inline int Simulate(int argc, const char** argv) {
  boost_vector_t xyz(3);
  xyz[0] = 1.0;
  xyz[1] = 1.0;
  xyz[2] = 1.0;

  typedef boost::numeric::odeint::rosenbrock4<double> ode_int;

  // set-up the Rosenbrock integrator
  auto stepper =
    boost::numeric::odeint::make_dense_output<ode_int>(1.e-6,1.e-6);

  // perform the time-integration
  integrate_const(
    stepper, std::make_pair(ODE_system(), ODE_jacobian()), xyz,
    0.0, 30.0, 0.01,
    ODE_output()
  );

  std::cout << "Simulation completed successfully!\n" << std::endl;
  return 0;
}

} // namespace lorenz

#endif // LORENZ_H_
