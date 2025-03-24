# Modified OpenCDA Core Modules (`opencda/`)

This folder contains a **modified version of OpenCDA's core modules**, including perception, planning, control, etc.

You should **place this folder inside a full OpenCDA project** (e.g., `...\openCDA\opencda`) in order to run simulations with CARLA.

---

## How to Run (Windows)

1. **Start CARLA**
Please follow the instruction on:  
[CARLA Setup Guide](https://carla.readthedocs.io/en/latest/build_faq/#cannot-run-example-scripts-or-runtimeerror-rpcrpc_error-during-call-in-function-version) to download CARLA Simulator and set up the environment.
   
   Run CARLA simulator:
   ```bash
   cd path/to/carla/root
   CarlaUE4.exe

2. **Activate Conda environment**
Please follow the instruction on:
[OpenCDA Documentation](https://opencda-documentation.readthedocs.io/en/latest/index.html) to download OpenCDA and set up the environment.

   Activate OpenCDA:
   ```bash
   activate opencda

3. **Run the simulation**

   ```bash
   cd ...\openCDA
   python opencda.py -t single_town_carla -v 0.9.13 --apply_ml

**Notes**

This folder (opencda/) must be placed under your OpenCDA project root for it to work correctly.

Make sure your environment includes required packages (carla, torch, etc.).

This modification likely affects core logic — ensure you keep backups of your original files if needed.








