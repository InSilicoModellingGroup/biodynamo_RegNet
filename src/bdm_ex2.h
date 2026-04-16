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
#ifndef BDM_EX2_H_
#define BDM_EX2_H_

#include "biodynamo.h"

#include "regulatory_network.h"

namespace bdm {

struct Lorenz_rhs_ {
  void operator()(const boost_vector_t& x, boost_vector_t& dxdt, double t, Agent* agent) const {
    dxdt[0] = sigma * x[1] - sigma * x[0];
    dxdt[1] = rho * x[0] - x[1] - x[0] * x[2];
    dxdt[2] = -beta * x[2] + x[0] * x[1];
  }
  // https://en.wikipedia.org/wiki/Lorenz_system
  double sigma = 10.0;
  double rho = 28.0;
  double beta = 8.0 / 3.0;
};

struct Lorenz_jac_ {
  void operator()(const boost_vector_t& x, boost_matrix_t& jac, double t, boost_vector_t& dfdt, Agent* agent) const {
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
    dfdt[0] = dfdt[1] = dfdt[2] = 0.0;
  }
  // https://en.wikipedia.org/wiki/Lorenz_system
  double sigma = 10.0;
  double rho = 28.0;
  double beta = 8.0 / 3.0;
};

struct Lorenz_out_ {
  void operator()(const boost_vector_t& x, real_t t, const Agent* agent) {
    auto& xyz = agent->GetPosition();
    std::clog << agent->GetUid()
              << ',' << xyz[0] << ',' << xyz[1] << ',' << xyz[2]
              << ',' << t << ',' << x[0] << ',' << x[1] << ',' << x[2];
    std::clog << std::endl;
  }
};

class Trajectory : public RegulatoryNetwork {
  BDM_BEHAVIOR_HEADER(Trajectory, RegulatoryNetwork, 1);

 public:
  Trajectory() { AlwaysCopyToNew(); }

  Trajectory(real_t dt, int n_dt, const std::vector<real_t>& x)
    : RegulatoryNetwork(dt, n_dt, x, ODE_solver::Rosenbrock,
                        Lorenz_rhs_(), Lorenz_jac_(), Lorenz_out_()) { trail_=0.0; }

  virtual ~Trajectory() = default;

  void Initialize(const NewAgentEvent& event) override {
    Base::Initialize(event);

    if (auto* other = dynamic_cast<Trajectory*>(event.existing_behavior)) {
      trail_ = other->trail_;
    } else {
      Log::Fatal("Trajectory::EventConstructor",
                 "other was not of type Trajectory");
    }
  }

  void Run(Agent* agent) override {
    Base::Run(agent);

    Real3 xyz;
    for (int i=0; i<3; i++)
      xyz[i] = this->GetSpecie(i);

    auto* cell = bdm_static_cast<Cell*>(agent);
    cell->SetPosition(xyz);
  }

 private:
   /// keep track of the trail of the agent
   real_t trail_;
};

namespace ex2 {

inline int Simulate(int argc, const char** argv) {
  // set-up the BioDynaMo simulation parameters
  auto set_parameters = [](Param* param) {
    param->use_progress_bar = false;
    param->bound_space = Param::BoundSpaceMode::kOpen;
    param->min_bound =    0.0;
    param->max_bound = +100.0;
    param->export_visualization = true;
    param->visualization_interval = 1;
    param->visualize_agents["Cell"] = { "diameter_", "volume_" };
    param->statistics = false;
    param->simulation_time_step = 1.0;
  };

  Simulation sim(argc, argv, set_parameters);

  // time-step of the regulatory network solver
  const real_t dt_RN = 0.01;

  {
    Real3 xyz{1.0, 1.0, 1.0};

    Cell* cell = new Cell();
    cell->SetDiameter(1.0);
    cell->SetPosition(xyz);
    cell->AddBehavior(new Trajectory(dt_RN, 10, {xyz[0], xyz[1], xyz[2]}));

    sim.GetExecutionContext()->AddAgent(cell);
  }

  for (int s=0; s<3000; s++)
    sim.GetScheduler()->Simulate(1);

  std::cout << "Simulation completed successfully!\n" << std::endl;
  return 0;
}

} // namespace ex2

} // namespace bdm

#endif // BDM_EX2_H_
