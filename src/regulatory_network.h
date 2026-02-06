// -----------------------------------------------------------------------------
//
// Copyright (C) 2021 CERN & University of Surrey for the benefit of the
// BioDynaMo collaboration. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// See the LICENSE file distributed with this work for details.
// See the NOTICE file distributed with this work for additional information
// regarding copyright ownership.
//
// -----------------------------------------------------------------------------

#ifndef REGULATORY_NETWORK_H_
#define REGULATORY_NETWORK_H_

#include <functional>
#include <vector>

#include "core/behavior/behavior.h"

#include "boost/numeric/odeint.hpp"
#include "boost/phoenix/core.hpp"
#include "boost/phoenix/operator.hpp"

typedef boost::numeric::ublas::vector<double>  boost_vector_t;
typedef boost::numeric::ublas::matrix<double>  boost_matrix_t;
typedef std::map<std::string, bdm::DiffusionGrid*>  std_map_bdm_dg_t;

namespace bdm {

enum class ODE_solver {
  Rosenbrock,
  RungeKutta
};

class RegulatoryNetwork : public Behavior {
  BDM_BEHAVIOR_HEADER(RegulatoryNetwork, Behavior, 1);

 public:
  RegulatoryNetwork() { AlwaysCopyToNew(); }

  RegulatoryNetwork(real_t dt, const std::vector<real_t>& x, ODE_solver m,
      const std::function<void(const boost_vector_t&, boost_vector_t&, real_t)>& rhs,
      const std::function<void(const boost_vector_t&, boost_matrix_t&, real_t, boost_vector_t&)>& jacob,
      const std::function<void(const boost_vector_t&, real_t)>& out) {
    AlwaysCopyToNew();
    SetInitialSpecies(x);
    time_step_ = dt;
    rhs_ = rhs;
    jacob_ = jacob;
    out_ = out;
    method_ = m;
  }

  virtual ~RegulatoryNetwork() = default;

  void Initialize(const NewAgentEvent& event) override {
    Base::Initialize(event);

    if (auto* r = dynamic_cast<RegulatoryNetwork*>(event.existing_behavior)) {
      current_time_ = r->current_time_;
      time_step_ = r->time_step_;
      current_species_ = r->current_species_;
      previous_species_ = r->previous_species_;
      //
      rhs_ = r->rhs_;
      jacob_ = r->jacob_;
      out_ = r->out_;
      method_ = r->method_;
    } else {
      Log::Fatal("RegulatoryNetwork::EventConstructor",
                 "other was not of type RegulatoryNetwork");
    }
  }

  const size_t GetNumberOfSpecies() const { return current_species_.size(); }
  const boost_vector_t& GetSpecies() const { return current_species_; }
  const real_t& GetSpecie(size_t i) const { return current_species_[i]; }

  /// Method Run() contains the implementation for Runge-Khutta and Euler
  /// methods for solving ODE(s).
  void Run(Agent* agent) override {
    // update the previous solution
    previous_species_ = current_species_;
    // initialize the time-integration scheme
    if (ODE_solver::Rosenbrock == method_) {
      auto stepper =
        boost::numeric::odeint::make_dense_output<boost::numeric::odeint::rosenbrock4<double>>(1.e-6,1.e-6);
      // perform time integration
      integrate_const(
        stepper, std::make_pair(rhs_ , jacob_), current_species_,
        current_time_, current_time_+time_step_, time_step_/1000, out_
      );
    } else if (ODE_solver::RungeKutta == method_) {
      auto stepper =
        boost::numeric::odeint::make_dense_output<boost::numeric::odeint::runge_kutta_dopri5<boost_vector_t>>(1.e-6,1.e-6);
      // perform time integration
      integrate_const(
        stepper, rhs_, current_species_,
        current_time_, current_time_+time_step_, time_step_/1000, out_
      );
    } else {
      Log::Fatal("RegulatoryNetwork::Run",
                 "invalid type of ODE solution method indicated");
    }
    // update the time of the regulatory network
    current_time_ += time_step_;
  }

 protected:

  void SetInitialSpecies(const std::vector<real_t>& x) {
    const size_t n_species = x.size();
    //
    current_species_.resize(n_species);
    previous_species_.resize(n_species);
    for (size_t i=0; i<n_species; i++)
      current_species_[i] = previous_species_[i] = x[i];
  }

 private:
  /// Pseudo-time for ODE(s) time integration
  real_t current_time_ = 0.0;
  /// Time-step for ODE(s) time integration
  real_t time_step_ = 1.0;
  /// Current solution of the species concentration
  boost_vector_t current_species_ = {};
  /// Previous solution of the species concentration
  boost_vector_t previous_species_ = {};
  /// Method used for the ODE(s) numerical solution
  ODE_solver method_;

  std::function<void(const boost_vector_t&, boost_vector_t&, real_t)> rhs_;
  std::function<void(const boost_vector_t&, boost_matrix_t&, real_t, boost_vector_t&)> jacob_;
  std::function<void(const boost_vector_t&, real_t)> out_;
};

}  // namespace bdm

#endif  // REGULATORY_NETWORK_H_
