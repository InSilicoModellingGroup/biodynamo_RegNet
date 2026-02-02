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

class ODE_parameters {
 public:
  ODE_parameters(real_t a, real_t b, real_t c)
  : alpha_(a), beta_(b), gamma_(c) {}
  ODE_parameters(const ODE_parameters& p) {
    alpha_ = p.alpha(); beta_ = p.beta(); gamma_ = p.gamma();
  }

  inline real_t alpha() const { return alpha_; }
  inline real_t beta()  const { return beta_; }
  inline real_t gamma() const { return gamma_; }

 private:
  real_t alpha_, beta_, gamma_;
};

struct ODE_system : public ODE_parameters {
  ODE_system(Cell* cell, const ODE_parameters& p, const std_map_bdm_dg_t& dg_map)
  : ODE_parameters(p), c_(cell), dg_(dg_map) {}

  void operator()(const boost_vector_t& x, boost_vector_t& dxdt, real_t t) const {
    auto& crd = c_->GetPosition();
    auto& dg = dg_.find("cytokine")->second;
    const real_t ecm_c = dg->GetValue(crd);

    dxdt[0] = alpha() * x[1];
    dxdt[1] = beta() * x[1] - beta() * x[1] * x[0] * x[0] - gamma() * x[0];
  }

 private:
  Cell* c_;
  std_map_bdm_dg_t dg_;
};

struct ODE_jacobian : public ODE_parameters {
  ODE_jacobian(Cell* cell, const ODE_parameters& p, const std_map_bdm_dg_t& dg_map)
  : ODE_parameters(p), c_(cell), dg_(dg_map) {}

  void operator()(const boost_vector_t& x, boost_matrix_t& jac, real_t t, boost_vector_t& dfdt) const {
    jac(0, 0) = 0.0;
    jac(0, 1) = alpha();
    jac(1, 0) = -2.0 * beta() * x[0] * x[1] - gamma();
    jac(1, 1) = beta() - beta() * x[0] * x[0];
    //
    dfdt[0] = 0.0;
    dfdt[1] = 0.0;
  }

 private:
  Cell* c_;
  std_map_bdm_dg_t dg_;
};

struct ODE_output {
  ODE_output(Cell* cell, const std_map_bdm_dg_t& dg_map)
  : c_(cell), dg_(dg_map) {}

  void operator()(const boost_vector_t& x, real_t t) {
    auto& crd = c_->GetPosition();

    std::cout << ' ' << c_->GetUid() << std::flush;
    std::cout << ':' << crd[0] << ' ' << crd[1] << ' ' << crd[2] << std::flush;
    std::cout << ':' << t << std::flush;
    std::cout << ':' << x[0] << ' ' << x[1] << std::endl;
  }

 private:
  Cell* c_;
  std_map_bdm_dg_t dg_;
};

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

  // time-step of the BioDynaMo simulator
  const real_t dt_BDM = sim.GetParam()->simulation_time_step;
  // time-step of the regulatory network solver
  const real_t dt_RN = 1000.0;
  // parameters of the regulatory network
  const ODE_parameters rn_p(1.e+0, 1.e+3, 1.e+0);
  // BioDynaMo's diffusion grid sample points in each dimension
  int n_DG = 51;

  ModelInitializer::DefineSubstance(kCytokine, "cytokine", 0./dt_BDM, 0./dt_BDM, n_DG);
  ModelInitializer::AddBoundaryConditions(
      kCytokine, BoundaryConditionType::kNeumann,
      std::make_unique<ConstantBoundaryCondition>(0)
    );

  std::map<std::string, DiffusionGrid*> dg_m;
  dg_m.insert(std::make_pair("cytokine", rm->GetDiffusionGrid("cytokine")));

  auto* cell = new Cell({0.01, 0.02, 0.03});
  cell->SetDiameter(1.0);
  cell->AddBehavior(new RegulatoryNetwork(dt_RN, {1., 1.},
                                          ODE_system(cell,rn_p,dg_m),
                                          ODE_jacobian(cell,rn_p,dg_m),
                                          ODE_output(cell,dg_m)));

  rm->AddAgent(cell);

  sim.GetScheduler()->Simulate(1);

  std::cout << "Simulation completed successfully!\n" << std::endl;
  return 0;
}

}  // namespace bdm

#endif // BIODYNAMO_REGNET_H_
