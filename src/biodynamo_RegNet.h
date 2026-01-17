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

inline void vdp_rhs(const boost_vector_t& x, boost_vector_t& dxdt, real_t t) {
  const double mu = 1000.0;
  //
  dxdt[0] = x[1];
  dxdt[1] = mu * x[1] - mu * x[1] * x[0] * x[0] - x[0];
}

inline void vdp_jacob (const boost_vector_t& x, boost_matrix_t& jac, real_t t, boost_vector_t& dfdt) {
  const double mu = 1000.0;
  //
  jac(0, 0) = 0.0;
  jac(0, 1) = 1.0;
  jac(1, 0) = -2.0 * mu * x[0] * x[1] - 1.0;
  jac(1, 1) = mu - mu * x[0] * x[0];
  //
  dfdt[0] = 0.0;
  dfdt[1] = 0.0;
}

inline void vdp_out(const boost_vector_t& x, real_t t) {
  std::cout << ' ' << t << std::flush;
  std::cout << ':' << x[0] << ' ' << x[1] << std::endl;
}

inline int Simulate(int argc, const char** argv) {
  // https://biodynamo.github.io/api/structbdm_1_1Param.html
  auto set_parameters = [](Param* param) {
    param->use_progress_bar = false;
    param->bound_space = Param::BoundSpaceMode::kClosed;
    param->min_bound = -10.0;
    param->max_bound = +10.0;
    param->export_visualization = true;
    param->visualization_interval = 1;
    param->visualize_agents["Cell"] = { "diameter_", "volume_" };
    param->statistics = false;
    param->simulation_time_step = 1.0;
    param->visualize_diffusion = { Param::VisualizeDiffusion{"cytokine", true, true} };
    param->calculate_gradients = false;
    param->diffusion_method = "euler";
  };

  Simulation sim(argc, argv, set_parameters);

  auto* rm = sim.GetResourceManager();

  // agent-based model simulation time-step
  const real_t dt_BDM = sim.GetParam()->simulation_time_step;
  // regulatory network simulation time-step
  const real_t dt_RN = 1000.0;
  // BioDynaMo's diffusion grid sample points in each dimension
  int n_DG = 51;

  ModelInitializer::DefineSubstance(kCytokine, "cytokine", 0./dt_BDM, 0./dt_BDM, n_DG);
  ModelInitializer::AddBoundaryConditions(
      kCytokine, BoundaryConditionType::kNeumann,
      std::make_unique<ConstantBoundaryCondition>(0)
    );

  auto* cell = new Cell({0., 0., 0.});
  cell->SetDiameter(1.0);
  cell->AddBehavior(new RegulatoryNetwork(dt_RN, {1., 1.}, vdp_rhs, vdp_jacob, vdp_out));

  rm->AddAgent(cell);

  sim.GetScheduler()->Simulate(10);

  std::cout << "Simulation completed successfully!\n" << std::endl;
  return 0;
}

}  // namespace bdm

#endif // BIODYNAMO_REGNET_H_
