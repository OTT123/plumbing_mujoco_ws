#include "simulate_engine/simulate_engine.hpp"
#include <iostream>
#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

void mycontroller(const mjModel *m, mjData *d) {
  // printouts for debugging purposes
  std::cout << "number of position coordinates: " << m->nq << std::endl;
  std::cout << "number of degrees of freedom: " << m->nv << std::endl;
  std::cout << "joint position: " << d->qpos[0] << std::endl;
  std::cout << "joint velocity: " << d->qvel[0] << std::endl;
  std::cout << "Sensor output: " << d->sensordata[0] << std::endl;

  for (auto i = 0ul; i < m->nv; ++i) {
    d->ctrl[i] = 100;
  }
}

int main(int argc, char const **argv) {
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have different versions");
  }
  const char *filename = nullptr;
  if (argc > 1) {
    filename = argv[1];
  }
  
  char *mj_plugin_dir = "mujoco_plugin";
  MujocoSimulateEngine mujoco_sim(MUJOCO_PLUGIN_DIR);
  mjcb_control = mycontroller;

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mujoco::Simulate>(
      std::make_unique<mujoco::GlfwAdapter>(), mujoco_sim.GetMujocoCameraPtr(),
      mujoco_sim.GetMujocoOptionPtr(), mujoco_sim.GetMujocoPerturbPtr(),
      /* is_passive = */ false);
  // start physics thread
  std::thread physicsthreadhandle(&MujocoSimulateEngine::PhysicsThread,
                                  &mujoco_sim, sim.get(), filename);

  // start simulation UI loop (blocking call)
  sim->RenderLoop();
  physicsthreadhandle.join();
  return 0;
}
