#include "simulate_engine/simulate_engine.hpp"

MujocoSimulateEngine::MujocoSimulateEngine(char *mujoco_plugin_dir) {
  mujoco_plugin_dir_ = mujoco_plugin_dir;
  mjv_defaultCamera(&cam_);
  mjv_defaultOption(&opt_);
  mjv_defaultPerturb(&pert_);
  scanPluginLibraries();
}

std::string MujocoSimulateEngine::getExecutableDir() {
  constexpr char kPathSep = '/'; // linux
  const char *path = "/proc/self/exe";
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == -1) {
        if (errno == EINVAL) {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: "
                  << strerror(errno) << '\n';
        return "";
      } else {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();
  if (realpath.empty()) {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}
void MujocoSimulateEngine::scanPluginLibraries() {
  // check and print plugins that are linked directly into the executable
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i = 0; i < nplugin; ++i) {
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    }
  }
  const std::string sep = "/"; // linux
  // try to open the ${EXECDIR}/MUJOCO_PLUGIN_DIR directory
  // ${EXECDIR} is the directory containing the simulate binary itself
  // MUJOCO_PLUGIN_DIR is the MUJOCO_PLUGIN_DIR preprocessor macro
  const std::string executable_dir = getExecutableDir();
  if (executable_dir.empty()) {
    return;
  }

  const std::string plugin_dir = getExecutableDir() + sep + mujoco_plugin_dir_;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char *filename, int first, int count) {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        }
      });
}
const char *MujocoSimulateEngine::Diverged(int disableflags, const mjData *d) {
  if (disableflags & mjDSBL_AUTORESET) {
    for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
      if (d->warning[w].number > 0) {
        return mju_warningText(w, d->warning[w].lastinfo);
      }
    }
  }
  return nullptr;
}
mjModel *MujocoSimulateEngine::LoadModel(const char *file,
                                         mujoco::Simulate &sim) {
  // this copy is needed so that the mujoco::sample_util::strlen call below
  // compiles
  char filename[mujoco::Simulate::kMaxFilenameLength];
  mujoco::sample_util::strcpy_arr(filename, file);

  // make sure filename is not empty
  if (!filename[0]) {
    return nullptr;
  }

  // load and compile
  char loadError[kErrorLength] = "";
  mjModel *mnew = 0;
  auto load_start = mujoco::Simulate::Clock::now();
  if (mujoco::sample_util::strlen_arr(filename) > 4 &&
      !std::strncmp(filename + mujoco::sample_util::strlen_arr(filename) - 4,
                    ".mjb",
                    mujoco::sample_util::sizeof_arr(filename) -
                        mujoco::sample_util::strlen_arr(filename) + 4)) {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      mujoco::sample_util::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);

    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mujoco::sample_util::strlen_arr(loadError);
      if (loadError[error_length - 1] == '\n') {
        loadError[error_length - 1] = '\0';
      }
    }
  }
  auto load_interval = mujoco::Simulate::Clock::now() - load_start;
  double load_seconds = Seconds(load_interval).count();

  if (!mnew) {
    std::printf("%s\n", loadError);
    mujoco::sample_util::strcpy_arr(sim.load_error, loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n",
                loadError);
    sim.run = 0;
  }

  // if no error and load took more than 1/4 seconds, report load time
  else if (load_seconds > 0.25) {
    mujoco::sample_util::sprintf_arr(loadError, "Model loaded in %.2g seconds",
                                     load_seconds);
  }

  mujoco::sample_util::strcpy_arr(sim.load_error, loadError);

  return mnew;
}
void MujocoSimulateEngine::PhysicsLoop(mujoco::Simulate &sim) {
  // cpu-sim syncronization point
  std::chrono::time_point<mujoco::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      mjModel *mnew = LoadModel(sim.dropfilename, sim);
      sim.droploadrequest.store(false);

      mjData *dnew = nullptr;
      if (mnew)
        dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.dropfilename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d_);
        mj_deleteModel(m_);

        m_ = mnew;
        d_ = dnew;
        mj_forward(m_, d_);

      } else {
        sim.LoadMessageClear();
      }
    }

    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      sim.LoadMessage(sim.filename);
      mjModel *mnew = LoadModel(sim.filename, sim);
      mjData *dnew = nullptr;
      if (mnew)
        dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.filename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d_);
        mj_deleteModel(m_);

        m_ = mnew;
        d_ = dnew;
        mj_forward(m_, d_);

      } else {
        sim.LoadMessageClear();
      }
    }

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery
    //  life
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

      // run only if model is present
      if (m_) {
        // running
        if (sim.run) {
          bool stepped = false;

          // record cpu time at start of iteration
          const auto startCPU = mujoco::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d_->time - syncSim;

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger
          // than syncmisalign
          bool misaligned = std::abs(Seconds(elapsedCPU).count() / slowdown -
                                     elapsedSim) > kSyncMisalign_;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 ||
              syncCPU.time_since_epoch().count() == 0 || misaligned ||
              sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d_->time;
            sim.speed_changed = false;

            // run single step, let next iteration deal with timing
            mj_step(m_, d_);
            const char *message = Diverged(m_->opt.disableflags, d_);
            if (message) {
              sim.run = 0;
              mujoco::sample_util::strcpy_arr(sim.load_error, message);
            } else {
              stepped = true;
            }
          }

          // in-sync: step until ahead of cpu
          else {
            bool measured = false;
            mjtNum prevSim = d_->time;

            double refreshTime = kSimRefreshFraction_ / sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((d_->time - syncSim) * slowdown) <
                       mujoco::Simulate::Clock::now() - syncCPU &&
                   mujoco::Simulate::Clock::now() - startCPU <
                       Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measured_slowdown =
                    std::chrono::duration<double>(elapsedCPU).count() /
                    elapsedSim;
                measured = true;
              }

              // inject noise
              sim.InjectNoise();

              // call mj_step
              mj_step(m_, d_);
              const char *message = Diverged(m_->opt.disableflags, d_);
              if (message) {
                sim.run = 0;
                mujoco::sample_util::strcpy_arr(sim.load_error, message);
              } else {
                stepped = true;
              }

              // break if reset
              if (d_->time < prevSim) {
                break;
              }
            }
          }

          // save current state to history buffer
          if (stepped) {
            sim.AddToHistory();
          }
        }

        // paused
        else {
          // run mj_forward, to update rendering and joint sliders
          mj_forward(m_, d_);
          sim.speed_changed = true;
        }
      }
    } // release std::lock_guard<std::mutex>
  }
}
void MujocoSimulateEngine::PhysicsThread(mujoco::Simulate *sim,
                                         const char *filename) {
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr) {
    sim->LoadMessage(filename);
    m_ = LoadModel(filename, *sim);
    if (m_) {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      d_ = mj_makeData(m_);
    }
    if (d_) {
      sim->Load(m_, d_, filename);

      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      mj_forward(m_, d_);

    } else {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  mj_deleteData(d_);
  mj_deleteModel(m_);
}
