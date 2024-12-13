#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include "mujoco/mujoco.h"
#include "mujoco/simulate/array_safety.h"
#include "mujoco/simulate/glfw_adapter.h"
#include "mujoco/simulate/simulate.h"

// for linux
#include <sys/errno.h>
#include <unistd.h>
#define kErrorLength 1024
using Seconds = std::chrono::duration<double>;

class MujocoSimulateEngine {
protected:
  char *mujoco_plugin_dir_;
  // constants
  const double kSyncMisalign_ =
      0.1; // maximum mis-alignment before re-sync (simulation seconds)
  const double kSimRefreshFraction_ =
      0.7; // fraction of refresh available for simulation
  mjModel *m_ = nullptr;
  mjData *d_ = nullptr;
  mjvCamera cam_;
  mjvOption opt_;
  mjvPerturb pert_;

  std::string getExecutableDir();
  void scanPluginLibraries();
  const char *Diverged(int disableflags, const mjData *d);
  mjModel *LoadModel(const char *file, mujoco::Simulate &sim);
  void PhysicsLoop(mujoco::Simulate &sim);

public:
  MujocoSimulateEngine(char *mujoco_plugin_dir);
  void PhysicsThread(mujoco::Simulate *sim, const char *filename);

  mjModel *GetMujocoModelPtr() { return m_; };
  mjData *GetMujocoDataPtr() { return d_; };
  mjvCamera* GetMujocoCameraPtr(){return &cam_;};
  mjvOption* GetMujocoOptionPtr(){return &opt_;};
  mjvPerturb* GetMujocoPerturbPtr(){return &pert_;};
  // ~MujocoSimulateEngine();
};
