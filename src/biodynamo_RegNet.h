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
#ifndef BIODYNAMO_REGNET_H_
#define BIODYNAMO_REGNET_H_

#include "biodynamo.h"

#include "regulatory_network.h"

namespace bdm {

enum Substances { kCytokine };

// Van der Pol oscillator
//     https://juliareach.github.io/ReachabilityAnalysis.jl/dev/generated_examples/VanDerPol/#Van-der-Pol-oscillator
//
// d[x0]/d[t] = alpha * x1
// d[x1]/d[t] = beta * x1 * (1-x0^2) - gamma * x0

struct ODE_system {
  Agent* agent;
  const std::map<std::string, DiffusionGrid*>& mdg;
  const std::vector<real_t> param;

  ODE_system(std::map<std::string, DiffusionGrid*>& m, const std::vector<real_t>& p)
  : agent(0), mdg(m), param(p) { }

  void operator()(const boost_vector_t& x, boost_vector_t& dxdt, real_t t) const {
    // auto& xyz = agent.GetPosition();
    auto dg = mdg.find("cytokine")->second;
    const real_t cytokine = dg->GetValue({0,0,0});

    dxdt[0] = t * x[0] + param[0];
    dxdt[1] = t * x[1] * x[1] + param[1];
    dxdt[2] = t + x[2] + param[2];
  }
};

struct ODE_jacobian {
  Agent* agent;
  const std::map<std::string, DiffusionGrid*>& mdg;
  const std::vector<real_t> param;

  ODE_jacobian(std::map<std::string, DiffusionGrid*>& m, const std::vector<real_t>& p)
  : agent(0), mdg(m), param(p) { }

  void operator()(const boost_vector_t& x, boost_matrix_t& jac, real_t t, boost_vector_t& dfdt) const {
    // auto& xyz = agent.GetPosition();
    auto dg = mdg.find("cytokine")->second;
    const real_t cytokine = dg->GetValue({0,0,0});

    jac(0, 0) = t;
    jac(0, 1) = 0.0;
    jac(0, 2) = 0.0;
    jac(1, 0) = 0.0;
    jac(1, 1) = t * 2.0 * x[1];
    jac(1, 2) = 0.0;
    jac(2, 0) = 0.0;
    jac(2, 1) = 0.0;
    jac(2, 2) = 1.0;
    //
    dfdt[0] = 0.0;
    dfdt[1] = 0.0;
    dfdt[2] = 0.0;
  }
};

struct ODE_output {
  void operator()(const boost_vector_t& x, real_t t, const Agent* a) {
    std::cout << a->GetUid() << std::flush;
    std::cout << " : " << a->GetPosition() << std::flush;
    std::cout << " : " << t << std::flush;
    std::cout << " : " << x[0] << ' ' << x[1] << ' ' << x[2] << std::endl;
  }
};

inline int Simulate(int argc, const char** argv) {
  // https://biodynamo.github.io/api/structbdm_1_1Param.html
  auto set_parameters = [](Param* param) {
    param->use_progress_bar = false;
    param->bound_space = Param::BoundSpaceMode::kClosed;
    param->min_bound = -10.0;
    param->max_bound = +10.0;
    param->export_visualization = false;
    // param->export_visualization = true;
    // param->visualization_interval = 1;
    // param->visualize_agents["Cell"] = { "diameter_", "volume_" };
    param->statistics = false;
    param->simulation_time_step = 1.0;
    // param->visualize_diffusion = { Param::VisualizeDiffusion{"cytokine", true, true} };
    // param->calculate_gradients = false;
    // param->diffusion_method = "euler";
  };

  Simulation sim(argc, argv, set_parameters);

  auto* rm = sim.GetResourceManager();

  // time-step of the BioDynaMo simulator
  const real_t dt_BDM = sim.GetParam()->simulation_time_step;
  // time-step of the regulatory network solver
  const real_t dt_RN = 0.01;
  // BioDynaMo's diffusion grid sample points in each dimension
  int n_DG = 51;

  ModelInitializer::DefineSubstance(kCytokine, "cytokine", 0./dt_BDM, 0./dt_BDM, n_DG);
  ModelInitializer::AddBoundaryConditions(
      kCytokine, BoundaryConditionType::kNeumann,
      std::make_unique<ConstantBoundaryCondition>(0)
    );

  std::map<std::string, DiffusionGrid*> dg_map;
  dg_map.insert(std::make_pair("cytokine", rm->GetDiffusionGrid("cytokine")));

  auto* cell = new Cell({0.01, 0.02, 0.03});
  cell->SetDiameter(30);
  cell->SetAdherence(0.4);
  cell->SetMass(1.0);
  cell->AddBehavior(new RegulatoryNetwork(dt_RN, 100, {1., 5., 7.},
                                          //ODE_solver::Euler,
                                          //ODE_solver::Rosenbrock,
                                          ODE_solver::RungeKutta,
                                          ODE_system(dg_map,{0.2,0.0,3.0}),
                                          ODE_jacobian(dg_map,{0.2,0.0,3.0}),
                                          ODE_output()));
  // add this cell into the simulation
  rm->AddAgent(cell);

  for (int s=0; s<10; s++)
    sim.GetScheduler()->Simulate(1);

  std::cout << "Simulation completed successfully!\n" << std::endl;
  return 0;
}

}  // namespace bdm

#endif // BIODYNAMO_REGNET_H_
