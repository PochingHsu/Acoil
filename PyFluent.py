import numpy
import ansys.fluent.core as pyfluent
import os
import time
import shutil
from func import res_csv, u_csv, res_summary, max_vel_summary

GCI = 0  # GCI report
solver_session = pyfluent.launch_fluent(precision="double",version="2d", processor_count=1, show_gui=True)
solver_session.file.read_case(file_name="A-coil1-Finer_07.cas.h5")  # put the CFDmodel (.h5) file in the same folder
# Generate boundary conditions
FR = numpy.array([470, 510, 550, 590, 630, 670, 700, 710, 750, 760, 790, 820, 830, 870, 880, 910,
                  940, 950, 990, 1000, 1030, 1060, 1070, 1100, 1120, 1180, 1200, 1240, 1300, 1360,
                 1400, 1420, 1480, 1540, 1600]) # CFM
# FR = numpy.array([1150]) # CFM
fluid_density = 1.225
convert_factor = 0.00047194745
inlet_height = 0.21
inlet_area = 0.21*0.41
mesh = 'Finer_07'
for i in FR:
    # Calculate MFR in 2D flow simulation
    MFR = i*convert_factor*fluid_density*inlet_height*1/inlet_area/2  # divide 2 for symmetry, divide inlet_area for 2D
    print('mass_flow: %0.4f [kg/s]' % MFR)
    # model setup
    solver_session.setup.boundary_conditions.mass_flow_inlet['inlet'](mass_flow=MFR)
    solver_session.solution.initialization.hybrid_initialize()
    solver_session.tui.solve.set.number_of_iterations("4000")
    # Run solver
    solver_session.solution.run_calculation.iterate()
    solver_session.tui.file.read_journal('journal.jou.txt')
    solver_session.tui.solve.set.number_of_iterations("1")
    solver_session.solution.run_calculation.iterate()
    time.sleep(3)
    # output velocity profile
    new_name = "hx_inlet_u_" + str(i) + '_' +mesh
    os.rename("myResult", new_name)
    # output residual
    new_name_res = "hx_inlet_res_" + str(i) + '_' +mesh
    shutil.copy('res', 'res_copy')
    os.rename('res_copy', new_name_res)
acoil = 'A-coil1'
path = r'C:\Users\pochi\PycharmProjects\AcoilCFDSuggorate'  # use your path
# output residual as csv
Mesh = res_csv(acoil, mesh)
# output velocity as csv
Mesh = u_csv(acoil, mesh)
# residual summary file
res_summary(acoil, Mesh, path)
if GCI:
    # max velocity summary file (for GCI)
    max_vel_summary(acoil, path)
