"""
Milk-Run routing strategy: Terminal -> Multiple Power Plants (2+) -> Terminal

3-Tier Priority System for Ship Routing (REVISED):

PRINCIPLE 1 (HIGHEST PRIORITY - Base Strategy):
- ALL ships visit at least 2 plants (Milk-Run delivery) - MANDATORY
- Shinju ships PRIORITIZED (smallest capacity = lowest OPEX)
- Keep M[k] ≤ capacity_ship[k] → slack_meq[k] = 0 → cost savings
- Goal: slack_meq = 0, slack_time = 0 for all ships
- This is the FOUNDATION of routing strategy

PRINCIPLE 2 (Upgrade to Larger Ship):
- Applied when Principle 1 (Milk-Run + Shinju) causes Utilization > 100%
- Upgrade to larger ship: Shinju → WSD59 → Coral (automatic forced upgrade)
- MAINTAIN Milk-Run (still visit 2+ plants)
- Example: If Shinju Milk-Run needs 3,000 m³ (capacity 2,513) → Auto upgrade to WSD59
- Implementation: 
1. ship_used binary + priority constraints (smaller ships first)
2. Force Coral usage when 6+ WSD used (prevent WSD59 overload)
3. **Limit slack_meq ≤ 0.01 × capacity
- Result: No ship can exceed 101% utilization → Automatic upgrade to larger ship

PRINCIPLE 3 (Cost-Efficiency Check & Route Split) - ENHANCED WITH UPGRADE:
- After Principle 2, check if larger ship is cost-efficient:
  - Otherwise → SPLIT: use multiple ships, 1 plant each (Terminal → Plant → Terminal)
  - **NEW: After split, if utilization still > 100% → Upgrade to larger ship**
    - Example: Shinju split util > 100% → Use WSD59 split instead
    - Example: WSD59 split util > 100% → Use Coral split instead
- Choose more efficient option
- Result: Even split routes maintain utilization ≤ 100% through ship upgrade

Ship Capacity: Shinju (2,513 m³) < WSD59 (5,000 m³) < Coral (7,500 m³)
Ship Priority: Shinju (highest) > WSD59 > Coral (lowest)

Big-M Technique: Used for conditional constraints (e.g., "IF util > 100% THEN upgrade")

Algorithm: Outer Approximation (OA) with Single Tree Implementation
mip_solver='gurobi_persistent', nlp_solver='ipopt'

"""

# =================================================================
# Import Libraries
# =================================================================
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pyomo.environ as pyomo

# Path configuration
import os
import sys
conda_path = os.path.dirname(sys.executable)
library_bin = os.path.join(conda_path, "Library", "bin")
if library_bin not in os.environ["PATH"]:
    os.environ["PATH"] = library_bin + os.pathsep + os.environ["PATH"]

# Other imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("LNG Supply Chain to Power Plants in Eastern Indonesia")
print("with Varied Capacity Ship Model Routing using MINLP")
print("=" * 70)

# =================================================================
# Step 1: Create Model
# =================================================================
model = pyomo.ConcreteModel(name="LNG_Supply_")

# =================================================================
# Step 2: Define Sets
# =================================================================
model.i = pyomo.Set(initialize=['Donggi', 'Tangguh', 'Bontang'], doc='Terminals')
model.j = pyomo.Set(initialize=[
    'Ambon', 'Ternate', 'Namlea', 'Seram', 'Sorong', 'Manokwari',
    'Biak', 'Serui', 'Jayapura', 'Merauke', 'Timika', 'Nabire',
    'Dobo', 'Langgur', 'Saumlaki'
], doc='Power Plants')

# Full ship set
model.k = pyomo.Set(initialize=[
    'Shinju_1', 'WSD59_1', 'Coral_1', 'Shinju_2', 'WSD59_2', 'Coral_2', 
    'Shinju_3', 'WSD59_3', 'Coral_3', 'Shinju_4', 'WSD59_4', 'Coral_4', 
    'Shinju_5', 'WSD59_5', 'Coral_5', 'Shinju_6', 'WSD59_6', 'Coral_6', 
    'Shinju_7', 'WSD59_7', 'Coral_7', 'Shinju_8', 'WSD59_8', 'Coral_8', 
    'Shinju_9', 'WSD59_9', 'Coral_9', 'Shinju_10', 'WSD59_10', 'Coral_10'
], doc='Ships')

# =================================================================
# Step 3: Define Parameters
# =================================================================

# Distance between terminals and power plants (km) - u_t_p
u_t_p = { 
    ('Donggi', 'Ambon'): 877.85, ('Donggi', 'Ternate'): 588.94, ('Donggi', 'Namlea'): 637.09,
    ('Donggi', 'Seram'): 946.37, ('Donggi', 'Sorong'): 1105.64, ('Donggi', 'Manokwari'): 1496.42,
    ('Donggi', 'Biak'): 1646.43, ('Donggi', 'Serui'): 1770.51, ('Donggi', 'Jayapura'): 2235.36,
    ('Donggi', 'Merauke'): 2246.48, ('Donggi', 'Timika'): 1764.96, ('Donggi', 'Nabire'): 1766.81,
    ('Donggi', 'Dobo'): 1548.27, ('Donggi', 'Langgur'): 1348.26, ('Donggi', 'Saumlaki'): 1411.22,
    
    ('Tangguh', 'Ambon'): 783.40, ('Tangguh', 'Ternate'): 1009.34, ('Tangguh', 'Namlea'): 738.95,
    ('Tangguh', 'Seram'): 711.17, ('Tangguh', 'Sorong'): 448.18, ('Tangguh', 'Manokwari'): 838.96,
    ('Tangguh', 'Biak'): 988.97, ('Tangguh', 'Serui'): 1113.05, ('Tangguh', 'Jayapura'): 1577.90,
    ('Tangguh', 'Merauke'): 1635.32, ('Tangguh', 'Timika'): 1038.97, ('Tangguh', 'Nabire'): 1111.20,
    ('Tangguh', 'Dobo'): 825.99, ('Tangguh', 'Langgur'): 672.28, ('Tangguh', 'Saumlaki'): 1022.30,
    
    ('Bontang', 'Ambon'): 1761.25, ('Bontang', 'Ternate'): 1168.61, ('Bontang', 'Namlea'): 1505.68,
    ('Bontang', 'Seram'): 1829.78, ('Bontang', 'Sorong'): 1874.22, ('Bontang', 'Manokwari'): 2266.85,
    ('Bontang', 'Biak'): 2415.01, ('Bontang', 'Serui'): 2540.94, ('Bontang', 'Jayapura'): 2959.50,
    ('Bontang', 'Merauke'): 3052.10, ('Bontang', 'Timika'): 2539.09, ('Bontang', 'Nabire'): 2537.24,
    ('Bontang', 'Dobo'): 2322.41, ('Bontang', 'Langgur'): 2164.99, ('Bontang', 'Saumlaki'): 2102.02
}
model.u_t_p = pyomo.Param(model.i, model.j, initialize=u_t_p)

# Distance between power plants and power plants (km) - u_p_p
u_p_p = {
    ('Ambon', 'Ambon'): 0, ('Ambon', 'Ternate'): 913.04, ('Ambon', 'Namlea'): 611.16, 
    ('Ambon', 'Seram'): 150.01, ('Ambon', 'Sorong'): 796.36, ('Ambon', 'Manokwari'): 1187.13, 
    ('Ambon', 'Biak'): 1335.29, ('Ambon', 'Serui'): 1461.23, ('Ambon', 'Jayapura'): 1926.08, 
    ('Ambon', 'Merauke'): 1505.68, ('Ambon', 'Timika'): 1018.60, ('Ambon', 'Nabire'): 1457.52,
    ('Ambon', 'Dobo'): 761.17, ('Ambon', 'Langgur'): 575.97, ('Ambon', 'Saumlaki'): 792.66,
    
    ('Ternate', 'Ambon'): 913.04, ('Ternate', 'Ternate'): 0, ('Ternate', 'Namlea'): 463.00, 
    ('Ternate', 'Seram'): 979.71, ('Ternate', 'Sorong'): 818.58, ('Ternate', 'Manokwari'): 1211.21, 
    ('Ternate', 'Biak'): 1359.37, ('Ternate', 'Serui'): 1485.30, ('Ternate', 'Jayapura'): 1948.30, 
    ('Ternate', 'Merauke'): 2076.09, ('Ternate', 'Timika'): 1483.45, ('Ternate', 'Nabire'): 1481.60,
    ('Ternate', 'Dobo'): 1266.77, ('Ternate', 'Langgur'): 1107.50, ('Ternate', 'Saumlaki'): 1457.52,
    
    ('Namlea', 'Ambon'): 611.16, ('Namlea', 'Ternate'): 463.00, ('Namlea', 'Namlea'): 0, 
    ('Namlea', 'Seram'): 679.68, ('Namlea', 'Sorong'): 592.64, ('Namlea', 'Manokwari'): 983.41, 
    ('Namlea', 'Biak'): 1133.42, ('Namlea', 'Serui'): 1257.51, ('Namlea', 'Jayapura'): 1722.36, 
    ('Namlea', 'Merauke'): 1813.11, ('Namlea', 'Timika'): 1222.32, ('Namlea', 'Nabire'): 1253.80,
    ('Namlea', 'Dobo'): 1003.78, ('Namlea', 'Langgur'): 846.36, ('Namlea', 'Saumlaki'): 1144.54,
    
    ('Seram', 'Ambon'): 150.01, ('Seram', 'Ternate'): 979.71, ('Seram', 'Namlea'): 679.68, 
    ('Seram', 'Seram'): 0, ('Seram', 'Sorong'): 724.13, ('Seram', 'Manokwari'): 1116.76, 
    ('Seram', 'Biak'): 1264.92, ('Seram', 'Serui'): 1390.85, ('Seram', 'Jayapura'): 1855.70, 
    ('Seram', 'Merauke'): 1468.64, ('Seram', 'Timika'): 961.19, ('Seram', 'Nabire'): 1387.15,
    ('Seram', 'Dobo'): 703.76, ('Seram', 'Langgur'): 518.56, ('Seram', 'Saumlaki'): 744.50,
    
    ('Sorong', 'Ambon'): 796.36, ('Sorong', 'Ternate'): 818.58, ('Sorong', 'Namlea'): 592.64, 
    ('Sorong', 'Seram'): 724.13, ('Sorong', 'Sorong'): 0, ('Sorong', 'Manokwari'): 438.92, 
    ('Sorong', 'Biak'): 587.08, ('Sorong', 'Serui'): 713.02, ('Sorong', 'Jayapura'): 1176.02, 
    ('Sorong', 'Merauke'): 1648.28, ('Sorong', 'Timika'): 1051.94, ('Sorong', 'Nabire'): 709.32,
    ('Sorong', 'Dobo'): 837.10, ('Sorong', 'Langgur'): 685.24, ('Sorong', 'Saumlaki'): 1035.27,
    
    ('Manokwari', 'Ambon'): 1187.13, ('Manokwari', 'Ternate'): 1211.21, ('Manokwari', 'Namlea'): 983.41, 
    ('Manokwari', 'Seram'): 1116.76, ('Manokwari', 'Sorong'): 438.92, ('Manokwari', 'Manokwari'): 0, 
    ('Manokwari', 'Biak'): 224.09, ('Manokwari', 'Serui'): 331.51, ('Manokwari', 'Jayapura'): 809.32, 
    ('Manokwari', 'Merauke'): 2040.90, ('Manokwari', 'Timika'): 1444.56, ('Manokwari', 'Nabire'): 320.40,
    ('Manokwari', 'Dobo'): 1229.73, ('Manokwari', 'Langgur'): 1076.01, ('Manokwari', 'Saumlaki'): 1426.04,
    
    ('Biak', 'Ambon'): 1335.29, ('Biak', 'Ternate'): 1359.37, ('Biak', 'Namlea'): 1133.42, 
    ('Biak', 'Seram'): 1264.92, ('Biak', 'Sorong'): 587.08, ('Biak', 'Manokwari'): 224.09, 
    ('Biak', 'Biak'): 0, ('Biak', 'Serui'): 235.20, ('Biak', 'Jayapura'): 622.27, 
    ('Biak', 'Merauke'): 2189.06, ('Biak', 'Timika'): 1592.72, ('Biak', 'Nabire'): 353.73,
    ('Biak', 'Dobo'): 1377.89, ('Biak', 'Langgur'): 1224.17, ('Biak', 'Saumlaki'): 1574.20,
    
    ('Serui', 'Ambon'): 1461.23, ('Serui', 'Ternate'): 1485.30, ('Serui', 'Namlea'): 1257.51, 
    ('Serui', 'Seram'): 1390.85, ('Serui', 'Sorong'): 713.02, ('Serui', 'Manokwari'): 331.51, 
    ('Serui', 'Biak'): 235.20, ('Serui', 'Serui'): 0, ('Serui', 'Jayapura'): 559.30, 
    ('Serui', 'Merauke'): 2315.00, ('Serui', 'Timika'): 1718.66, ('Serui', 'Nabire'): 218.54,
    ('Serui', 'Dobo'): 1503.82, ('Serui', 'Langgur'): 1350.11, ('Serui', 'Saumlaki'): 1700.14,
    
    ('Jayapura', 'Ambon'): 1926.08, ('Jayapura', 'Ternate'): 1948.30, ('Jayapura', 'Namlea'): 1722.36, 
    ('Jayapura', 'Seram'): 1855.70, ('Jayapura', 'Sorong'): 1176.02, ('Jayapura', 'Manokwari'): 809.32, 
    ('Jayapura', 'Biak'): 622.27, ('Jayapura', 'Serui'): 559.30, ('Jayapura', 'Jayapura'): 0, 
    ('Jayapura', 'Merauke'): 2778.00, ('Jayapura', 'Timika'): 2183.51, ('Jayapura', 'Nabire'): 738.95,
    ('Jayapura', 'Dobo'): 1968.68, ('Jayapura', 'Langgur'): 1814.96, ('Jayapura', 'Saumlaki'): 2164.99,
    
    ('Merauke', 'Ambon'): 1505.68, ('Merauke', 'Ternate'): 2076.09, ('Merauke', 'Namlea'): 1813.11, 
    ('Merauke', 'Seram'): 1468.64, ('Merauke', 'Sorong'): 1648.28, ('Merauke', 'Manokwari'): 2040.90, 
    ('Merauke', 'Biak'): 2189.06, ('Merauke', 'Serui'): 2315.00, ('Merauke', 'Jayapura'): 2778.00, 
    ('Merauke', 'Merauke'): 0, ('Merauke', 'Timika'): 868.59, ('Merauke', 'Nabire'): 2311.30,
    ('Merauke', 'Dobo'): 992.67, ('Merauke', 'Langgur'): 1048.23, ('Merauke', 'Saumlaki'): 1037.12,
    
    ('Timika', 'Ambon'): 1018.60, ('Timika', 'Ternate'): 1483.45, ('Timika', 'Namlea'): 1222.32, 
    ('Timika', 'Seram'): 961.19, ('Timika', 'Sorong'): 1051.94, ('Timika', 'Manokwari'): 1444.56, 
    ('Timika', 'Biak'): 1592.72, ('Timika', 'Serui'): 1718.66, ('Timika', 'Jayapura'): 2183.51, 
    ('Timika', 'Merauke'): 868.59, ('Timika', 'Timika'): 0, ('Timika', 'Nabire'): 1714.95,
    ('Timika', 'Dobo'): 451.89, ('Timika', 'Langgur'): 496.34, ('Timika', 'Saumlaki'): 809.32,
    
    ('Nabire', 'Ambon'): 1457.52, ('Nabire', 'Ternate'): 1481.60, ('Nabire', 'Namlea'): 1253.80, 
    ('Nabire', 'Seram'): 1387.15, ('Nabire', 'Sorong'): 709.32, ('Nabire', 'Manokwari'): 320.40, 
    ('Nabire', 'Biak'): 353.73, ('Nabire', 'Serui'): 218.54, ('Nabire', 'Jayapura'): 738.95, 
    ('Nabire', 'Merauke'): 2311.30, ('Nabire', 'Timika'): 1714.95, ('Nabire', 'Nabire'): 0,
    ('Nabire', 'Dobo'): 1500.12, ('Nabire', 'Langgur'): 1346.40, ('Nabire', 'Saumlaki'): 1696.43,
    
    ('Dobo', 'Ambon'): 761.17, ('Dobo', 'Ternate'): 1266.77, ('Dobo', 'Namlea'): 1003.78, 
    ('Dobo', 'Seram'): 703.76, ('Dobo', 'Sorong'): 837.10, ('Dobo', 'Manokwari'): 1229.73, 
    ('Dobo', 'Biak'): 1377.89, ('Dobo', 'Serui'): 1503.82, ('Dobo', 'Jayapura'): 1968.68, 
    ('Dobo', 'Merauke'): 992.67, ('Dobo', 'Timika'): 451.89, ('Dobo', 'Nabire'): 1500.12,
    ('Dobo', 'Dobo'): 0, ('Dobo', 'Langgur'): 238.91, ('Dobo', 'Saumlaki'): 485.22,
    
    ('Langgur', 'Ambon'): 575.97, ('Langgur', 'Ternate'): 1107.50, ('Langgur', 'Namlea'): 846.36, 
    ('Langgur', 'Seram'): 518.56, ('Langgur', 'Sorong'): 685.24, ('Langgur', 'Manokwari'): 1076.01, 
    ('Langgur', 'Biak'): 1224.17, ('Langgur', 'Serui'): 1350.11, ('Langgur', 'Jayapura'): 1814.96, 
    ('Langgur', 'Merauke'): 1048.23, ('Langgur', 'Timika'): 496.34, ('Langgur', 'Nabire'): 1346.40,
    ('Langgur', 'Dobo'): 238.91, ('Langgur', 'Langgur'): 0, ('Langgur', 'Saumlaki'): 1016.75,
    
    ('Saumlaki', 'Ambon'): 792.66, ('Saumlaki', 'Ternate'): 1457.52, ('Saumlaki', 'Namlea'): 1144.54, 
    ('Saumlaki', 'Seram'): 744.50, ('Saumlaki', 'Sorong'): 1035.27, ('Saumlaki', 'Manokwari'): 1426.04, 
    ('Saumlaki', 'Biak'): 1574.20, ('Saumlaki', 'Serui'): 1700.14, ('Saumlaki', 'Jayapura'): 2164.99, 
    ('Saumlaki', 'Merauke'): 1037.12, ('Saumlaki', 'Timika'): 809.32, ('Saumlaki', 'Nabire'): 1696.43,
    ('Saumlaki', 'Dobo'): 485.22, ('Saumlaki', 'Langgur'): 1016.75, ('Saumlaki', 'Saumlaki'): 0
}
model.u_p_p = pyomo.Param(model.j, model.j, initialize=u_p_p, default=999999)

# LNG demand(j) at power plants (cubic meter) - demand
demand_data = {
    'Ambon': 125, 'Ternate': 181.6, 'Namlea': 52.83, 'Seram': 53.02,
    'Sorong': 241.37, 'Manokwari': 117.45, 'Biak': 74.06, 'Serui': 43.40,
    'Jayapura': 138.21, 'Merauke': 104.72, 'Timika': 188.52, 'Nabire': 23.11,
    'Dobo': 47.17, 'Langgur': 78.77, 'Saumlaki': 40.57
}
model.demand = pyomo.Param(model.j, initialize=demand_data, doc= 'LNG demand at power plant j (cubic meter)')

# Ship capacity ship(k)
capacity_ship_data = {
    'Shinju_1': 2513, 'WSD59_1': 5000, 'Coral_1': 7500, 'Shinju_2': 2513, 'WSD59_2': 5000,
    'Coral_2': 7500, 'Shinju_3': 2513, 'WSD59_3': 5000, 'Coral_3': 7500, 'Shinju_4': 2513,
    'WSD59_4': 5000, 'Coral_4': 7500, 'Shinju_5': 2513, 'WSD59_5': 5000, 'Coral_5': 7500,
    'Shinju_6': 2513, 'WSD59_6': 5000, 'Coral_6': 7500, 'Shinju_7': 2513, 'WSD59_7': 5000,
    'Coral_7': 7500, 'Shinju_8': 2513, 'WSD59_8': 5000, 'Coral_8': 7500, 'Shinju_9': 2513,
    'WSD59_9': 5000, 'Coral_9': 7500, 'Shinju_10': 2513, 'WSD59_10': 5000, 'Coral_10': 7500
}
model.capacity_ship = pyomo.Param(model.k, initialize=capacity_ship_data, doc='ship capacity (cubic meter)')

# Power plant storage capacity_pp(j)
capacity_pp_data = {
    'Ambon': 273.91, 'Ternate': 273.91, 'Namlea': 91.30, 'Seram': 182.61,
    'Sorong': 456.52, 'Manokwari': 182.61, 'Biak': 136.96, 'Serui': 91.30,
    'Jayapura': 365.22, 'Merauke': 182.61, 'Timika': 273.91, 'Nabire': 456.52,
    'Dobo': 91.30, 'Langgur': 182.61, 'Saumlaki': 91.30
}
model.capacity_pp = pyomo.Param(model.j, initialize=capacity_pp_data, doc=' LNG capacity at power plant j (cubic meter)')

# Ship speed(k) (knot)
ship_speed_data = {
    'Shinju_1': 24.076, 'WSD59_1': 25.928, 'Coral_1': 25.928, 'Shinju_2': 24.076, 'WSD59_2': 25.928,
    'Coral_2': 25.928, 'Shinju_3': 24.076, 'WSD59_3': 25.928, 'Coral_3': 25.928, 'Shinju_4': 24.076,
    'WSD59_4': 25.928, 'Coral_4': 25.928, 'Shinju_5': 24.076, 'WSD59_5': 25.928, 'Coral_5': 25.928,
    'Shinju_6': 24.076, 'WSD59_6': 25.928, 'Coral_6': 25.928, 'Shinju_7': 24.076, 'WSD59_7': 25.928,
    'Coral_7': 25.928, 'Shinju_8': 24.076, 'WSD59_8': 25.928, 'Coral_8': 25.928, 'Shinju_9': 24.076,
    'WSD59_9': 25.928, 'Coral_9': 25.928, 'Shinju_10': 24.076, 'WSD59_10': 25.928, 'Coral_10': 25.928
}
model.ship_speed = pyomo.Param(model.k, initialize=ship_speed_data, doc='average speed of ship (knot)')

# Fuel consumption(k) (tonnes/day)
fuel_consumption_data = {
    'Shinju_1': 7.7, 'WSD59_1': 16.5, 'Coral_1': 20.5, 'Shinju_2': 7.7, 'WSD59_2': 16.5,
    'Coral_2': 20.5, 'Shinju_3': 7.7, 'WSD59_3': 16.5, 'Coral_3': 20.5, 'Shinju_4': 7.7,
    'WSD59_4': 16.5, 'Coral_4': 20.5, 'Shinju_5': 7.7, 'WSD59_5': 16.5, 'Coral_5': 20.5,
    'Shinju_6': 7.7, 'WSD59_6': 16.5, 'Coral_6': 20.5, 'Shinju_7': 7.7, 'WSD59_7': 16.5,
    'Coral_7': 20.5, 'Shinju_8': 7.7, 'WSD59_8': 16.5, 'Coral_8': 20.5, 'Shinju_9': 7.7,
    'WSD59_9': 16.5, 'Coral_9': 20.5, 'Shinju_10': 7.7, 'WSD59_10': 16.5, 'Coral_10': 20.5
}
model.fuel_consumption = pyomo.Param(model.k, initialize=fuel_consumption_data, doc='fuel consumption (tonnes/day)')

# DWT (deadweight tonnage)
DWT_data = {
    'Shinju_1': 1150, 'WSD59_1': 3000, 'Coral_1': 3450, 'Shinju_2': 1150, 'WSD59_2': 3000,
    'Coral_2': 3450, 'Shinju_3': 1150, 'WSD59_3': 3000, 'Coral_3': 3450, 'Shinju_4': 1150,
    'WSD59_4': 3000, 'Coral_4': 3450, 'Shinju_5': 1150, 'WSD59_5': 3000, 'Coral_5': 3450,
    'Shinju_6': 1150, 'WSD59_6': 3000, 'Coral_6': 3450, 'Shinju_7': 1150, 'WSD59_7': 3000,
    'Coral_7': 3450, 'Shinju_8': 1150, 'WSD59_8': 3000, 'Coral_8': 3450, 'Shinju_9': 1150,
    'WSD59_9': 3000, 'Coral_9': 3450, 'Shinju_10': 1150, 'WSD59_10': 3000, 'Coral_10': 3450
}
model.DWT = pyomo.Param(model.k, initialize=DWT_data, doc='deadweight tonnage (metric tonnes)')

# Tug fee (USD/hour)
tug_fee_data = {
    'Shinju_1': 30.02, 'WSD59_1': 46.76, 'Coral_1': 46.76, 'Shinju_2': 30.02, 'WSD59_2': 46.76,
    'Coral_2': 46.76, 'Shinju_3': 30.02, 'WSD59_3': 46.76, 'Coral_3': 46.76, 'Shinju_4': 30.02,
    'WSD59_4': 46.76, 'Coral_4': 46.76, 'Shinju_5': 30.02, 'WSD59_5': 46.76, 'Coral_5': 46.76,
    'Shinju_6': 30.02, 'WSD59_6': 46.76, 'Coral_6': 46.76, 'Shinju_7': 30.02, 'WSD59_7': 46.76,
    'Coral_7': 46.76, 'Shinju_8': 30.02, 'WSD59_8': 46.76, 'Coral_8': 46.76, 'Shinju_9': 30.02,
    'WSD59_9': 46.76, 'Coral_9': 46.76, 'Shinju_10': 30.02, 'WSD59_10': 46.76, 'Coral_10': 46.76
}
model.tug_fee = pyomo.Param(model.k, initialize=tug_fee_data, doc='tug service cost (USD per hour)')

# Gross tonnage GT(k)
GT_data = {
    'Shinju_1': 2930, 'WSD59_1': 5832, 'Coral_1': 7833, 'Shinju_2': 2930, 'WSD59_2': 5832,
    'Coral_2': 7833, 'Shinju_3': 2930, 'WSD59_3': 5832, 'Coral_3': 7833, 'Shinju_4': 2930,
    'WSD59_4': 5832, 'Coral_4': 7833, 'Shinju_5': 2930, 'WSD59_5': 5832, 'Coral_5': 7833,
    'Shinju_6': 2930, 'WSD59_6': 5832, 'Coral_6': 7833, 'Shinju_7': 2930, 'WSD59_7': 5832,
    'Coral_7': 7833, 'Shinju_8': 2930, 'WSD59_8': 5832, 'Coral_8': 7833, 'Shinju_9': 2930,
    'WSD59_9': 5832, 'Coral_9': 7833, 'Shinju_10': 2930, 'WSD59_10': 5832, 'Coral_10': 7833
}
model.GT = pyomo.Param(model.k, initialize=GT_data, doc='gross tonnage')

# Scalars
model.loading_time = 6  # hours
model.unloading_time = 6  # hours
model.operational_days = 330  # days per year
model.fuel_fee = 448  # USD/tonne
model.volume_limit = 0.15  # upper limit for tank
model.port_time = 2  # hours
model.delivery_period = 7  # days (It can be changed)
model.harbor_fee = 0.007  # per GT
model.guide_fee = 0.0018  # per GT  

# =================================================================
# Step 4: Define Variables
# =================================================================

# Binary variables for Milk-Run routing (3 types for complete path)
model.x_t_p = pyomo.Var(model.i, model.j, model.k, domain=pyomo.Binary,
                        initialize=0, doc='1 if arc from terminal i to power plant j by ship k')
model.x_p_p = pyomo.Var(model.j, model.j, model.k, domain=pyomo.Binary,
                        initialize=0, doc='1 if arc between power plants j and j by ship k')
model.x_p_t = pyomo.Var(model.j, model.i, model.k, domain=pyomo.Binary,
                        initialize=0, doc='1 if arc from power plant j to terminal i by ship k')

# Integer variables
model.y_t = pyomo.Var(model.i, model.k, domain=pyomo.NonNegativeIntegers,
                      doc='Number of visits to terminal i by ship k')
model.y_p = pyomo.Var(model.j, model.k, domain=pyomo.NonNegativeIntegers,
                      doc='Number of visits to power plant j by ship k')

# Positive Variables
model.Tmr = pyomo.Var(model.k, domain=pyomo.NonNegativeReals,
                        initialize=0, doc='Milk-Run travel time')
model.Cfuel = pyomo.Var(model.k, domain=pyomo.NonNegativeReals, 
                        doc='Fuel cost for one trip')
model.DIST = pyomo.Var(model.k, domain=pyomo.NonNegativeReals,
                        initialize=0, doc='Total distance traveled by ship k')
model.Time = pyomo.Var(model.k, domain=pyomo.NonNegativeReals,
                       bounds=(0, model.delivery_period * 24), initialize=0, doc='Total time needed to supply LNG by ship k')
model.M = pyomo.Var(model.k, domain=pyomo.NonNegativeReals,
                        initialize=0, doc='Volume filled in each delivery by ship k')

# TANK with explicit bounds to prevent constraint violation 
# Lower bound: demand x delivery_period (minimum to satisfy demand)
# Upper bound: capacity_pp x delivery_period (maximum storage capacity)
def tank_bounds_rule(model,j):
    lb = model.demand[j] * model.delivery_period
    ub = model.capacity_pp[j] * model.delivery_period
    return (lb, ub)

def tank_init_rule(model, j):
    """Initialize TANK to lower bound (minimum required to meet demand)"""
    return model.demand[j] * model.delivery_period

model.TANK = pyomo.Var(model.j, domain=pyomo.NonNegativeReals,
                        bounds=tank_bounds_rule, initialize=tank_init_rule, 
                        doc='Volume of LNG tank at power plant j (with bounds)')
model.SHIP = pyomo.Var(model.k, domain=pyomo.NonNegativeReals,
                        initialize=0, doc='Rental price of ship k')

# Cost Variables
model.z = pyomo.Var(domain=pyomo.Reals, initialize=0, 
                        doc='Total cost needed for supplying LNG')
model.OPEX = pyomo.Var(domain=pyomo.Reals, initialize=0, 
                        doc='Total operating cost needed for LNG supply chain')
model.FUEL = pyomo.Var(model.k, domain=pyomo.Reals,
                        initialize=0, doc='Fuel consumption for every round trip')
model.TUG = pyomo.Var(model.k, domain=pyomo.Reals,
                        initialize=0, doc='Tug service cost')
model.HARBOR = pyomo.Var(model.k, domain=pyomo.Reals,
                        initialize=0, doc='Harbor and mooring cost')
model.GUIDE = pyomo.Var(model.k, domain=pyomo.Reals, 
                        initialize=0, doc='Guide service cost')

# Slack variables for feasibility
model.slack_meq = pyomo.Var(model.k, domain=pyomo.NonNegativeReals,
                            initialize=0, doc='Capacity slack')
model.slack_time = pyomo.Var(model.k, domain=pyomo.NonNegativeReals,
                            initialize=0, doc='Time slack')

# ========== NEW Binary Variables for 3-Tier Priority System ==========

# Binary variable to track which ship is assigned to serve each plant
# This is needed because y_p[j,k] counts arcs (not binary 0/1), so we need
# A separate binary variable to prevent duplicate plant visits by different ships
model.plant_assignment = pyomo.Var(model.j, model.k, domain=pyomo.Binary,
                                   initialize=0,
                                   doc='1 if ship k is assigned to serve plant j (prevents duplicate visits)')

# Binary variable: 1 if ship k is allowed to visit only 1 plant (Tier 2 exception)
model.allow_single_visit = pyomo.Var(model.k, domain=pyomo.Binary,
                                     initialize=0, doc='1 if ship k can visit only 1 plant')

# Binary variable: 1 if ship k is used (has any route)
model.ship_used = pyomo.Var(model.k, domain=pyomo.Binary,
                            initialize=0, doc='1 if ship k is used (ship upgade logic)')

# =================================================================
# Step 5: Define Constraints for Milk-Run Model
# =================================================================
    
# MILK-RUN Distance calculation
def distance_equation_rule(model, k):
    terminal_to_plant = sum(model.u_t_p[i,j] * model.x_t_p[i,j,k]
                           for i in model.i for j in model.j)
    plant_to_plant = sum(model.u_p_p[j,j1] * model.x_p_p[j,j1,k]
                        for j in model.j for j1 in model.j if j != j1)
    plant_to_terminal = sum(model.u_t_p[i,j] * model.x_p_t[j,i,k]
                           for j in model.j for i in model.i)
    return model.DIST[k] == terminal_to_plant + plant_to_plant + plant_to_terminal
model.distance_equation = pyomo.Constraint(model.k, rule=distance_equation_rule)

# (Equation: Ship OPEX)
def ship_opex_rule(model, k):
    return model.SHIP[k] == (1.178e-05 * (model.capacity_ship[k])**2 + 
                            1.549E+00 * model.capacity_ship[k] + 9.142E+03) * (model.Time[k] / 24)
model.ship_opex = pyomo.Constraint(model.k, rule=ship_opex_rule)

# (Equation: OPEX calculation)
def opex_equation_rule(model):
    total_costs = sum((model.TUG[k] + model.FUEL[k] + model.SHIP[k] +
                       model.HARBOR[k] + model.GUIDE[k]) * model.x_t_p[i,j,k]
                          for i in model.i for j in model.j for k in model.k)
    return model.OPEX == total_costs * model.operational_days / model.delivery_period
model.opex_equation = pyomo.Constraint(rule=opex_equation_rule)

# (Equation: Fuel cost calculation)
def fuel_equation_rule(model, k):
    return model.FUEL[k] == model.fuel_fee * model.fuel_consumption[k] * model.Time[k] / 24
model.fuel_equation = pyomo.Constraint(model.k, rule=fuel_equation_rule)

# (Equation: Tug cost)
def tug_cost_rule(model, k):
    return model.TUG[k] == model.tug_fee[k] * sum((model.port_time + model.loading_time + model.unloading_time) *
                                                model.y_p[j,k] for j in model.j)
model.tug_cost = pyomo.Constraint(model.k, rule=tug_cost_rule)

# (Equation: Harbor cost)
def harbor_cost_rule(model, k):
    return model.HARBOR[k] == model.harbor_fee * model.GT[k] * sum(model.y_p[j,k] for j in model.j)
model.harbor_cost = pyomo.Constraint(model.k, rule=harbor_cost_rule)

# (Equation: Guide cost)
def guide_cost_rule(model, k):
    return model.GUIDE[k] == model.guide_fee * model.GT[k] * sum(model.y_p[j,k] for j in model.j)
model.guide_cost = pyomo.Constraint(model.k, rule=guide_cost_rule)

# (Equation: Sea_Time) Travel time calculation
def sea_time_rule(model, k):
    return model.Tmr[k] == model.DIST[k] / model.ship_speed[k]
model.sea_time = pyomo.Constraint(model.k, rule=sea_time_rule)

# (Equation: Total_Time) Total time calculation
def total_time_rule(model, k):
    loading_time = sum((model.port_time + model.loading_time) * model.x_t_p[i,j,k]
                      for i in model.i for j in model.j)
    unloading_time = sum((model.port_time + model.unloading_time) * model.y_p[j,k]
                        for j in model.j)
    return model.Time[k] == model.Tmr[k] + loading_time + unloading_time
model.total_time = pyomo.Constraint(model.k, rule=total_time_rule)

# (Equation:Delivery capacity)
def delivery_capacity_rule(model, k):
    terminal_to_plant = sum(model.TANK[j] * model.x_t_p[i,j,k]
                           for i in model.i for j in model.j)
    plant_to_plant = sum(model.TANK[j1] * model.x_p_p[j,j1,k]
                        for j in model.j for j1 in model.j if j != j1)
    small_capacity = 0.01 * model.capacity_ship[k] * sum(model.x_p_t[j,i,k]
                                                          for j in model.j for i in model.i)
    return model.M[k] == terminal_to_plant + plant_to_plant + small_capacity
model.delivery_capacity = pyomo.Constraint(model.k, rule=delivery_capacity_rule)

# (Equation: TANK_filled) or Tank capacity limit
def tank_filled_rule(model, j):
    return model.TANK[j] <= model.delivery_period * model.capacity_pp[j]
model.tank_filled = pyomo.Constraint(model.j, rule=tank_filled_rule)

# (Equation: Meq) or Ship capacity constraint with slack
# If M(k) > capacity_ship(k), then slack_meq(k) > 0 and penalty is applied
def meq_rule(model, k):
    return model.M[k] - model.slack_meq[k] <= model.capacity_ship[k]
model.meq = pyomo.Constraint(model.k, rule=meq_rule)

# (Equation: Storage limit) 
def storage_limit_rule(model, j):
    return (model.TANK[j] - model.delivery_period * model.demand[j]) / \
           (model.capacity_pp[j] * model.delivery_period) >= model.volume_limit
model.storage_limit = pyomo.Constraint(model.j, rule=storage_limit_rule)

# ===== Flow Conservation Constraints==============================
# Flow conservation at power plants (const1)
def const1_rule(model, j, k):
    inflow = sum(model.x_t_p[i,j,k] for i in model.i) + \
             sum(model.x_p_p[j1,j,k] for j1 in model.j if j1 != j)
    outflow = sum(model.x_p_t[j,i,k] for i in model.i) + \
              sum(model.x_p_p[j,j1,k] for j1 in model.j if j1 != j)
    return inflow == outflow
model.const1 = pyomo.Constraint(model.j, model.k, rule=const1_rule)

# Flow conservation at terminals (const2)
def const2_rule(model, i):
    return sum(model.x_t_p[i,j,k] for j in model.j for k in model.k) == \
           sum(model.x_p_t[j,i,k] for j in model.j for k in model.k)
model.const2 = pyomo.Constraint(model.i, rule=const2_rule)

# Each ship should start from at most a terminal (const3)
def const3_rule(model, i, k):
    return sum(model.x_t_p[i,j,k] for j in model.j) <= 1
model.const3 = pyomo.Constraint(model.i, model.k, rule=const3_rule)

# Each ship should end at at a terminal (const4)
def const4_rule(model, i, k):
    return sum(model.x_p_t[j,i,k] for j in model.j) <= 1
model.const4 = pyomo.Constraint(model.i, model.k, rule=const4_rule)

# Link x and y variables for terminals (const6_t)
def const6_t_rule(model, i, k):
    return sum(model.x_t_p[i,j,k] for j in model.j) + \
           sum(model.x_p_t[j,i,k] for j in model.j) == model.y_t[i,k]
model.const6_t = pyomo.Constraint(model.i, model.k, rule=const6_t_rule)

# Link x and y variables for power plants (const6_p)
def const6_p_rule(model, j, k):
    return sum(model.x_t_p[i,j,k] for i in model.i) + \
           sum(model.x_p_t[j,i,k] for i in model.i) + \
           sum(model.x_p_p[j1,j,k] for j1 in model.j if j1 != j) == model.y_p[j,k]
model.const6_p = pyomo.Constraint(model.j, model.k, rule=const6_p_rule)

# Subtour elimination constraint (const7)
def const7_rule(model):
    return sum(model.x_p_p[j1,j2,k] for j1 in model.j for j2 in model.j 
               for k in model.k if j1 != j2) <= len(model.j) - 1
model.const7 = pyomo.Constraint(rule=const7_rule)

# Ensure demand is met at each power plant (const8)
def const8_rule(model, j):
    return model.TANK[j] >= model.delivery_period * model.demand[j]
model.const8 = pyomo.Constraint(model.j, rule=const8_rule)

# Limit terminal visits (const9_t)
def const9_t_rule(model, i, k):
    return model.y_t[i,k] <= 5
model.const9_t = pyomo.Constraint(model.i, model.k, rule=const9_t_rule)

# Limit power plant visits (const9_p)
def const9_p_rule(model, j, k):
    return model.y_p[j,k] <= 10
model.const9_p = pyomo.Constraint(model.j, model.k, rule=const9_p_rule)

# Limit ALL incoming connections to power plant (const9_tp) (MODIFIED to prevent duplicate visits)
def const9_tp_rule(model, j):
    incoming_from_terminal = sum(model.x_t_p[i,j,k] for i in model.i for k in model.k)
    incoming_from_plants = sum(model.x_p_p[j1,j,k] for j1 in model.j if j1 != j for k in model.k)
    return incoming_from_terminal + incoming_from_plants <= 1
model.const9_tp = pyomo.Constraint(model.j, rule=const9_tp_rule)

# Ensure travel time does not exceed the delivery period time limit (const10)
def const10_rule(model, k):
    return model.Time[k] - model.slack_time[k] <= model.delivery_period * 24
model.const10 = pyomo.Constraint(model.k, rule=const10_rule)

# Ensure every power plant is visited exactly once (const11)
def const11_rule(model, j):
    return sum(model.y_p[j,k] for k in model.k) >= 1
model.const11 = pyomo.Constraint(model.j, rule=const11_rule)

# Flow continuity - ensure flow continuity from terminal to plants
def flow_continuity_rule(model, j, k):
    from_terminal = sum(model.x_t_p[i,j,k] for i in model.i) # The sum of routes entering power plant j from the terminal (with values of 0 or 1)
    to_other_plant = sum(model.x_p_p[j, j1, k] for j1 in model.j if j1 != j) # The sum of routes leaving power plant j to another power plant j1 (with values 0 or 1)
    # If the route comes from the terminal (from_terminal=1), it must depart to another power plant (to_other_plant =1)
    # If from_terminal = 0, the constraint is automatically satisfied
    return from_terminal >= to_other_plant
model.flow_continuity = pyomo.Constraint(model.j, model.k, rule=flow_continuity_rule)

# Flow conservation - ships leaving a terminal must return
def flow_conservation_rule(model, i, k):
    outgoing = sum(model.x_t_p[i,j,k] for j in model.j)
    returning = sum(model.x_p_t[j,i,k] for j in model.j)
    return outgoing == returning
model.flow_conservation = pyomo.Constraint(model.i, model.k, rule=flow_conservation_rule)

# ========== Milk-Run Routing Constraints (3-PRINCIPLE PRIORITY SYSTEM) ==========

# PRINCIPLE 1: All ships visit 2+ plants (Milk-Run) + Shinju prioritized
# PRINCIPLE 2: If Utilization > 100% → Upgrade to larger ship (maintain Milk-Run)
# PRINCIPLE 3: Cost-efficiency check → Split if larger ship inefficient

# ===== Big-M Value Explanation =====
# Big-M is a technical parameter for "switching" constraints on/off, NOT a cost penalty.
#
# How Big-M works:
#   M[k] <= capacity + BigM * allow_single_visit[k]
#   - If allow_single_visit = 0: M[k] <= capacity (constraint ACTIVE)
#   - If allow_single_visit = 1: M[k] <= capacity + BigM (constraint INACTIVE)
#
# Big-M sizing:
#   - Too small: Constraint doesn't fully deactivate → optimization fails
#   - Too large: Numerical instability → solver errors
#   - Optimal: Slightly larger than M[k] maximum possible value
#
# M[k] maximum calculation:
#   - M[k] = Total volume delivered by ship k in one trip
#   - Maximum = All power plants' total demand in delivery_period
#   - Total demand = Σ(demand[j] * delivery_period) ≈ 1,758 m³/day × 20 days = 35,160 m³
#
# BigM_util = 50,000:
#   ✓ Larger than M[k]_max (35,160) → constraints can be fully deactivated
#   ✓ Not excessively large → maintains numerical stability
#   ✓ Independent of ship capacity (Shinju: 2,513, WSD59: 5,000, Coral: 7,500)
#   ✓ Does NOT affect cost optimization (only constraint activation logic)
#
BigM_util = 50000  # Technical parameter for conditional constraints

# ===== PRINCIPLE 1: Base Milk-Run Constraint =====
# ALL ships must visit at least 2 plants (Milk-Run delivery)
# Exception: allow_single_visit[k] = 1 (Principle 3: route split)
def principle1_milkrun_base_rule(model, k):
    """
    Logic:
    - If allow_single_visit[k] = 0: visited_plants >= 2 * ship_used (Milk-Run enforced)
    - If allow_single_visit[k] = 1: visited_plants >= 2 * ship_used - BigM (no constraint, Tier 2)

    This ensures Milk-Run is the FOUNDATION strategy
    
    PRINCIPLE1: All ships visit at least 2 plants, unless allow_single_visit=1
    """
    ship_used = sum(model.x_t_p[i,j,k] for i in model.i for j in model.j)
    
    visited_plants = sum(
        sum(model.x_t_p[i,j,k] for i in model.i) +
        sum(model.x_p_p[j1,j,k] for j1 in model.j if j1 != j)
        for j in model.j
    )
    
    # Milk-Run enforced unless allow_single_visit = 1
    return visited_plants >= 2 * ship_used - BigM_util * model.allow_single_visit[k]

model.principle1_milkrun_base = pyomo.Constraint(model.k, rule=principle1_milkrun_base_rule)

# Upper bound: if allow_single_visit = 0, then M[k] < capacity_ship[k]
def principle3_split_upper_rule(model, k):
    """
    If allow_single_visit[k] = 0 (Milk-Run enforced):
    - M[k] <= capacity + tolerance (allows slight flexibility near 100%)
    
    Tolerance = 5% of capacity creates "gray zone" where either choice is valid
    This prevents infeasibility when M[k] ≈ capacity (Period 10, 15 issue)
    
    Original: M[k] <= capacity (too strict) and apply Enhanced one
    Enhanced: M[k] <= capacity * 1.05 (5% buffer for numerical stability)
    ENHANCED: Added tolerance buffer to avoid strict binary choice conflict
    
    PRINCIPLE 3: Upper bound for route split (5% tolerance buffer)
    """
    tolerance = 0.05 * model.capacity_ship[k]  # 5% tolerance buffer
    return model.M[k] <= model.capacity_ship[k] + tolerance + BigM_util * model.allow_single_visit[k]

model.principle3_split_upper = pyomo.Constraint(model.k, rule=principle3_split_upper_rule)

# Lower bound: if allow_single_visit = 1, then M[k] >= capacity_ship[k]
def priniciple3_split_lower_rule(model, k):
    """
    If allow_single_visit[k] = 1 (Route split allowed):
    - M[k] >= capacity - tolerance (allows slight flexibility near 100%)
    
    Tolerance = 5% of capacity creates "gray zone" where either choice is valid
    This enables cost comparison (upgrade vs split) without forcing strict binary choice
    
    Original: M[k] >= capacity (too strict)
    Enhanced: M[k] >= capacity * 0.95 (5% buffer for numerical stability)
    ENHANCED: Added tolerance buffer to avoid strict binary choice conflict

    PRINCIPLE 3: Lower bound for route split (5% tolerance buffer)
    """
    tolerance = 0.05 * model.capacity_ship[k]  # 5% tolerance buffer
    return model.M[k] >= model.capacity_ship[k] - tolerance - BigM_util * (1 - model.allow_single_visit[k])

model.principle3_split_lower = pyomo.Constraint(model.k, rule=priniciple3_split_lower_rule)

# ===== PRINCIPLE 2: Ship Usage/Selection Priority Constraints =====
# Smaller ships must be used before larger ships
# This ensures automatic upgrade when utilization > 100%

# Link ship_used to actual route usage
def ship_used_link_rule(model, k):
    """
    PRINCIPLE 2: Link ship_used binary variable to actual route usage.
    """
    # Link ship_used to actual route usage
    route_count = sum(model.x_t_p[i,j,k] for i in model.i for j in model.j)
    # ship_used[k] >= route_count ensures ship_used = 1 if any route exists
    # ship_used[k] is binary, so if route_count > 0, ship_used must be 1
    return model.ship_used[k] >= route_count

model.ship_used_link = pyomo.Constraint(model.k, rule=ship_used_link_rule)

# PRINCIPLE 2: Capacity-based Ship priority constraints
# Force smaller ships to be used before larger ships
# When smaller ships reach high utilization (near capacity), allow larger ships

def principle2_shinju_priority_rule(model, k):
    """
CAPACITY-BASED constraint with ADAPTIVE DYNAMIC BUFFER: Total WSD59 capacity limited by Shinju capacity.
# Considers total m³ capacity, not ship count
# Buffer adapts based on delivery_period for OPTIMAL solution
# Suitable for all demand scenarios (delivery_period 7-20 days)
    
Formula: WSD59_total_capacity <= Shinju_total_capacity * 1.5 + adaptive_buffer
    
    Adaptive buffer coefficients (ENHANCED for strict 100% utilization):
    - delivery_period ≤ 7:  0.61 × total_demand
    - delivery_period ≤ 10: 0.60 × total_demand
    - delivery_period ≤ 15: 0.60 × total_demand 
    - delivery_period ≥ 20: 0.59 × total_demand 
    
# Reason: Strict slack limit (1%) requires larger buffers to maintain feasibility
# Counter-intuitive: Smaller period → less demand → tighter constraint → LARGER buffer needed
    
    Example buffers by delivery_period:
    - 7 days (~10,569 m³):  buffer = 6,447 m³  (61% coefficient)
    - 10 days (~15,098 m³): buffer = 9,058 m³ (60% coefficient)
    - 15 days (~22,647 m³): buffer = 13,588 m³ (60% coefficient)
    - 20 days (~30,196 m³): buffer = 17,815 m³ (59% coefficient)

    Ratio 1.5: Allows WSD59 to provide 50% more capacity than Shinju
    Adaptive coefficients: Balance feasibility and ship priority across different demand levels
    PRINCIPLE 2: Shinju priority (WSD59 capacity limited by Shinju capacity with adaptive buffer)
    """
    if k.startswith('WSD59'): # WSD59 ships (string comparison)
        # Get ship lists
        shinju_ships = [ship for ship in model.k if ship.startswith('Shinju')]
        wsd59_ships = [ship for ship in model.k if ship.startswith('WSD59')]
        
        # Calculate TOTAL CAPACITY (m³) - not ship count!
        shinju_capacity = sum(model.ship_used[k_s] * 2513 for k_s in shinju_ships)  # Shinju: 2,513 m³
        wsd59_capacity = sum(model.ship_used[k_w] * 5000 for k_w in wsd59_ships)    # WSD59: 5,000 m³

        # Calculate total demand for dynamic buffer
        total_demand = sum(model.delivery_period * model.demand[j] for j in model.j) 
        
        # ADAPTIVE DYNAMIC BUFFER: Larger coefficients for smaller periods
        # Counter-intuitive logic: Smaller period → less demand → fewer ships → tighter constraint
        # Therefore: Smaller period needs LARGER buffer coefficient for flexibility
        # ENHANCED: Increased buffers for ALL periods to compensate for strict slack limit (1%)
        if model.delivery_period <= 7:
            wsd59_buffer = total_demand * 0.61   # 61%
        elif model.delivery_period <= 10:
            wsd59_buffer = total_demand * 0.60  # 60%
        elif model.delivery_period <= 15:
            wsd59_buffer = total_demand * 0.60  # 60%
        else:  # delivery_period >= 20
            wsd59_buffer = total_demand * 0.59  # 59%

        if k in []:
            return pyomo.Constraint.Skip
        else:
            # CAPACITY-BASED: WSD59 capacity <= Shinju capacity * ratio + dynamic buffer
            # Ratio 1.5 allows generous WSD59 usage for high demand
            # Adaptive ENHANCED buffer (period-specific, increased for strict 100% utilization):
            return wsd59_capacity <= shinju_capacity * 1.5 + wsd59_buffer
    else:
        return pyomo.Constraint.Skip
    
model.principle2_shinju_priority = pyomo.Constraint(model.k, rule=principle2_shinju_priority_rule)

def principle2_wsd59_priority_rule(model, k):
    """
    CAPACITY-BASED constraint with ADAPTIVE DYNAMIC BUFFER: Total Coral capacity limited by WSD59 capacity.
    - Considers total m³ capacity, not ship count
    - Buffer adapts based on delivery_period for OPTIMAL solution
    - More restrictive than Shinju→WSD59 (Coral is last resort)

    Formula: Coral_total_capacity <= WSD59_total_capacity * 1.2 + adaptive_buffer
    
   ADAPTIVE buffer coefficients (ENHANCED for strict 100% utilization):
    - period ≤ 7:  0.50 × total_demand (increased from 0.50, MORE RESTRICTIVE than WSD59: 0.61)
    - period ≤ 10: 0.50 × total_demand (increased from 0.50, MORE RESTRICTIVE than WSD59: 0.60)
    - period ≤ 15: 0.49 × total_demand (increased from 0.49, MORE RESTRICTIVE than WSD59: 0.60)
    - period ≥ 20: 0.47 × total_demand (increased from 0.47, MORE RESTRICTIVE than WSD59: 0.59)
    - Priority gap: ~0.15 (maintains ship hierarchy across all periods)

    Reason: Strict slack limit (1%) requires larger buffers to maintain feasibility
    Counter-intuitive: Smaller period → less demand → tighter constraint → LARGER buffer needed

    Example buffers by delivery_period:
    - 7 days (~10,569 m³):  buffer = 5,284 m³  (50% coefficient)
    - 10 days (~15,098 m³): buffer = 7,549 m³  (50% coefficient)
    - 15 days (~22,647 m³): buffer = 11,097 m³ (49% coefficient)
    - 20 days (~30,196 m³): buffer = 14,192 m³ (47% coefficient)

    Ratio 1.2: More restrictive than WSD59 (1.5), ensures Coral is last resort
    """
    if k.startswith('Coral'): # Coral ships (string comparison)
        # Get ship lists
        wsd59_ships = [ship for ship in model.k if ship.startswith('WSD59')]
        coral_ships = [ship for ship in model.k if ship.startswith('Coral')]
        
        # Calculate TOTAL CAPACITY (m³) - not ship count!
        wsd59_capacity = sum(model.ship_used[k_w] * 5000 for k_w in wsd59_ships)    # WSD59: 5,000 m³
        coral_capacity = sum(model.ship_used[k_c] * 7500 for k_c in coral_ships)    # Coral: 7,500 m³

        # Calculate total demand for dynamic buffer
        total_demand = sum(model.delivery_period * model.demand[j] for j in model.j)
        
        # ADAPATIVE DYNAMIC BUFFER: Larger coefficients for smaller periods
        # Coral MORE RESTRICTIVE than WSD59 (maintains ~0.15 gap across all periods)
        # Counter-intuitive logic: Smaller period needs LARGER buffer coefficient
        # Increased buffers for ALL periods to compensate for strict slack limit (1%)
        if model.delivery_period <= 7:
            coral_buffer = total_demand * 0.50  # 50% - (WSD59: 0.61)
        elif model.delivery_period <= 10:
            coral_buffer = total_demand * 0.50  # 50% - (WSD59: 0.60)
        elif model.delivery_period <= 15:
            coral_buffer = total_demand * 0.49  # 49% - (WSD59: 0.60)
        else:  # delivery_period >= 20
            coral_buffer = total_demand * 0.47  # 47% - (WSD59: 0.59)

        if k in []:
            return pyomo.Constraint.Skip
        else:
            # CAPACITY-BASED: Coral capacity <= WSD59 capacity * ratio + dynamic buffer
            # Ratio 1.2 is more restrictive than WSD59 constraint (1.5)
            # Adaptive ENHANCED buffer (period-specific, increased for strict 100% utilization):
            # - period 7: 50% → Large buffer, still restricts Coral (WSD59: 61%)
            # - period 10: 50% → Large buffer (WSD59: 60%)
            # - period 15: 49% → Medium buffer (WSD59: 60%)
            # - period 20: 47% → User confirmed (WSD59: 59%)
            # Maintains ship priority: Shinju → WSD59 → Coral
            return coral_capacity <= wsd59_capacity * 1.2 + coral_buffer 
    else:
        return pyomo.Constraint.Skip

model.principle2_wsd59_priority = pyomo.Constraint(model.k, rule=principle2_wsd59_priority_rule)

# ===== PRINCIPLE 2: ENHANCED - Limit slack_meq to maintain utilization ≤ 100% =====

def principle2_limit_slack_rule(model, k):
    """
    This forces model to select larger ships instead of excessive slack.
    Example:
    - Shinju (2,513 m³): max 25 m³ slack → max 101% utilization
    - WSD59 (5,000 m³):  max 50 m³ slack → max 101% utilization
    - Coral (7,500 m³):  max 75 m³ slack → max 101% utilization
    
    If route needs more → Must upgrade to larger ship (Shinju → WSD59 → Coral)
    
    Utilization MUST be ≤ 100% (strict limit, no overload allowed)
    Solution: Limit slack_meq <= 0.01 × capacity_ship[k] (max 1% tolerance → 101% max util)
    Result: Strict capacity enforcement + automatic ship upgrade when needed

    PRINCIPLE 2 ENHANCED: Limit slack_meq to maintain utilization ≤ 101% (forces ship upgrade when needed)
    """
    return model.slack_meq[k] <= 0.01 * model.capacity_ship[k]

model.principle2_limit_slack = pyomo.Constraint(model.k, rule=principle2_limit_slack_rule)

# ===== PRINCIPLE 3: ENHANCED - Route Split with Ship Upgrade =====
# After route split, if utilization still > 100%, use larger ship
# Example: Shinju split (1 route) util > 100% → WSD59 split (1 route)

def principle3_split_upgrade_shinju_rule(model, k):
    """
    For Shinju ships with single visit (allow_single_visit=1):
    - If M[k] > capacity (util > 100% even after split)
    
    This ensures: If Shinju split is insufficient → WSD59 split is used
    Example: Heavy route split into Shinju_1 (single) + WSD59_1 (single)
    
    Logic: slack_meq[k] > 0 means this Shinju needs more capacity than it has.
            If single visit AND has slack → Force WSD59 to also use single visit
    PRINCIPLE 3 ENHANCED: If Shinju split route needs upgrade, force WSD59 usage 
    """
    if k.startswith('Shinju'):
        wsd59_ships = [ship for ship in model.k if ship.startswith('WSD59')]
        wsd59_single_visits = sum(model.allow_single_visit[k_w] for k_w in wsd59_ships)

        # If this Shinju has single visit (allow_single_visit[k]=1) AND uses slack (capacity exceeded)
        # Then at least one WSD59 must have single visit (wsd59_single_visits >= 1)
        # Using BigM: slack_meq[k] * allow_single_visit[k] <= BigM * wsd59_single_visits
        # If allow_single_visit[k]=1 and slack_meq[k]>0 → wsd59_single_visits must be >= 1
        return model.slack_meq[k] * model.allow_single_visit[k] <= BigM_util * wsd59_single_visits
    else:
        return pyomo.Constraint.Skip

model.principle3_split_upgrade_shinju = pyomo.Constraint(model.k, rule=principle3_split_upgrade_shinju_rule)

def principle3_split_upgrade_wsd59_rule(model, k):
    """
    For WSD59 ships with single visit (allow_single_visit=1):
    - If M[k] > capacity (util > 100% even after split)
    - Then at least one Coral ship must also use single visit
    
    This ensures: If WSD59 split is insufficient → Coral split is used
    Example: Very heavy route split into WSD59_1 (single) + Coral_1 (single)
    
    Logic: slack_meq[k] > 0 means this WSD59 needs more capacity than it has.
            If single visit AND has slack → Force Coral to also use single visit
     PRINCIPLE 3 ENHANCED: WSD59 route split upgrade to Coral if needed.
    
    """
    if k.startswith('WSD59'):
        coral_ships = [ship for ship in model.k if ship.startswith('Coral')]
        coral_single_visits = sum(model.allow_single_visit[k_c] for k_c in coral_ships)

        # If WSD59 has single visit (allow_single_visit[k]=1) AND uses slack (capacity exceeded)
        # Then at least one Coral must have single visit (coral_single_visits >= 1)
        # Using BigM: slack_meq[k] * allow_single_visit[k] <= BigM * coral_single_visits
        # If allow_single_visit[k]=1 and slack_meq[k]>0 → coral_single_visits must be >= 1
        return model.slack_meq[k] * model.allow_single_visit[k] <= BigM_util * coral_single_visits
    else:
        return pyomo.Constraint.Skip

model.principle3_split_upgrade_wsd59 = pyomo.Constraint(model.k, rule=principle3_split_upgrade_wsd59_rule)

# - slack_meq penalty (500,000) + limit forces upgrade to larger ships
# - Ship usage priority (Principle 2) enforces Shinju → WSD59 → Coral order
# - Route split with ship upgrade ensures utilization ≤ 100% even for single-plant routes
# - Model automatically selects appropriate ship size for each route demand
 
# =================================================================
# Step 6: Define Objective Function
# =================================================================

# Minimize total operational cost + penalty for slack variables
def objective_rule(model):
    penalty_meq = 500000 * sum(model.slack_meq[k] for k in model.k)
    penalty_time = 500000 * sum(model.slack_time[k] for k in model.k)
    return model.OPEX + penalty_meq + penalty_time

model.objective = pyomo.Objective(rule=objective_rule, sense=pyomo.minimize)

# =================================================================
# Step 7: Solve Model with MindtPy (Milk-Run Optimization)
# =================================================================

print("\n" + "="*70)
print("STEP 7: SOLVING WITH MindtPy - UTILIZATION-BASED ROUTING")
print("="*70)

# Display problem size
print(f"\n--- Problem Size ---")
print(f"Terminals: {len(model.i)} ({', '.join(model.i)})")
print(f"Power Plants: {len(model.j)}")
print(f"Ships: {len(model.k)} (Coral: 10, WSD59: 10, Shinju: 10)")

# Calculate total demand
total_demand = sum(model.delivery_period * model.demand[j] for j in model.j)
print(f"\nTotal demand to satisfy: {total_demand:,.0f} cbm")

# Calculate total capacity
total_capacity = sum(model.capacity_ship[k] for k in model.k)
print(f"Total ship capacity available: {total_capacity:,.0f} cbm")

# Configure MindtPy solver
print("\n--- MindtPy Solver Configuration ---")
print("Algorithm: LP/NLP Based Branch-and-Bound (Single-tree OA)")
print("MIP Solver: gurobi_persistent")
print("NLP Solver: IPOPT")
print("Initial Strategy: None (cold start)")
print("Objective: Minimize total operational cost (OPEX + penalties)")
print("\nSolver Parameters:")
print("  - strategy: OA (Outer Approximation)")
print("  - single_tree: True (Branch-and-bound)")
print("  - integer_tolerance: 1e-4")
print("  - constraint_tolerance: 1e-5")

# Solve with MindtPy
start_time = time.time()

solver = SolverFactory('mindtpy')
results = solver.solve(
    model,
    strategy='OA',  # Outer Approximation
    mip_solver='gurobi_persistent',
    nlp_solver='ipopt',
    tee=True,
    single_tree=True,  # Enable single-tree implementation (LP/NLP Branch-and-Bound)
    init_strategy='rNLP',
    feasibility_norm='L_infinity',
    obj_bound=1e6,
    time_limit=1000,
    iteration_limit=1000,
    mip_solver_mipgap=0.05,
    integer_tolerance=1e-4,
    constraint_tolerance=1e-5,
)
solve_time = time.time() - start_time

# =================================================================
# Step 8: Display Results
# =================================================================

print("\n" + "="*70)
print("STEP 8: OPTIMIZATION RESULTS")
print("="*70)

# Check solve status
from pyomo.opt import SolverStatus, TerminationCondition

# Helper function to safely get variable values
def safe_value(var, default=0):
    """Safely get value from a Pyomo variable, returning default if uninitialized"""
    try:
        if var.value is None:
            return default
        return pyomo.value(var)
    except (ValueError, AttributeError):
        return default

if results.solver.termination_condition in [pyomo.TerminationCondition.optimal,
                                            pyomo.TerminationCondition.feasible]:
    print(f"Solution time: {solve_time:.2f} seconds")
    
    # ========== Display: Objective Function ==========
    print("\n" + "="*70)
    print("DISPLAY z.l (Objective Function)")
    print("="*70)
    obj_value = pyomo.value(model.objective)
    print(f"z = ${obj_value:,.2f}")

    # ========== Display: OPEX and Cost Components ==========
    print("\n" + "="*70)
    print("DISPLAY OPEX.l, FUEL.l, TUG.l, HARBOR.l, GUIDE.l, SHIP.l")
    print("="*70)

    if model.OPEX.value is not None:
        print(f"\nOPEX = ${pyomo.value(model.OPEX):,.2f}")
    else:
        print("\nOPEX = N/A (not calculated)")

    print("\nCost Components by Ship:")
    print(f"{'Ship':<15} {'FUEL($)':<15} {'TUG($)':<15} {'HARBOR($)':<15} {'GUIDE($)':<15} {'SHIP($)':<15}")
    print("-" * 90)

    active_ships = []
    for k in model.k:
        if pyomo.value(model.M[k]) > 0.1: # Only show active ships
            active_ships.append(k)
            fuel_val = pyomo.value(model.FUEL[k]) if model.FUEL[k].value else 0
            tug_val = pyomo.value(model.TUG[k]) if model.TUG[k].value else 0
            harbor_val = pyomo.value(model.HARBOR[k]) if model.HARBOR[k].value else 0
            guide_val = pyomo.value(model.GUIDE[k]) if model.GUIDE[k].value else 0
            ship_val = pyomo.value(model.SHIP[k]) if model.SHIP[k].value else 0

            print(f"{k:<15} {fuel_val:<15,.2f} {tug_val:<15,.2f} {harbor_val:<15,.2f} {guide_val:<15,.2f} {ship_val:<15,.2f}")

    # Totals
    total_fuel = sum(pyomo.value(model.FUEL[k]) for k in model.k if model.FUEL[k].value is not None)
    total_tug = sum(pyomo.value(model.TUG[k]) for k in model.k if model.TUG[k].value is not None)
    total_harbor = sum(pyomo.value(model.HARBOR[k]) for k in model.k if model.HARBOR[k].value is not None)
    total_guide = sum(pyomo.value(model.GUIDE[k]) for k in model.k if model.GUIDE[k].value is not None)
    total_ship = sum(pyomo.value(model.SHIP[k]) for k in model.k if model.SHIP[k].value is not None)

    print("-" * 90)
    print(f"{'TOTAL':<15} {total_fuel:<15,.2f} {total_tug:<15,.2f} {total_harbor:<15,.2f} {total_guide:<15,.2f} {total_ship:<15,.2f}")

    # ========== Display: Binary Variables (Routing) ==========
    print("\n" + "="*70)
    print("DISPLAY x_t_p.l, x_p_p.l, x_p_t.l (Binary Variables - Routes)")
    print("="*70)

    print("\nx_t_p.l (Terminal → Power Plant):")
    has_x_t_p = False
    for i in model.i:
        for j in model.j:
            for k in model.k:
                if safe_value(model.x_t_p[i,j,k]) > 0.5:
                    print(f"  {i} → {j} by {k} = 1")
                    has_x_t_p = True
    if not has_x_t_p:
        print("  (No routes)")

    print("\nx_p_p.l (Power Plant → Power Plant - Milk-Run):")
    has_x_p_p = False
    for j1 in model.j:
        for j2 in model.j:
            if j1 != j2:
                for k in model.k:
                    if safe_value(model.x_p_p[j1,j2,k]) > 0.5:
                        print(f"  {j1} → {j2} by {k} = 1")
                        has_x_p_p = True
    if not has_x_p_p:
        print("  (No inter-plant routes - All direct delivery)")

    print("\nx_p_t.l (Power Plant → Terminal - Return):")
    has_x_p_t = False
    for j in model.j:
        for i in model.i:
            for k in model.k:
                if safe_value(model.x_p_t[j,i,k]) > 0.5:
                    print(f"  {j} → {i} by {k} = 1")
                    has_x_p_t = True
    if not has_x_p_t:
        print("  (No return routes)")

    # ========== Display: Integer Variables (Visits) ==========
    print("\n" + "="*70)
    print("DISPLAY y_t.l, y_p.l (Integer Variables - Visit Counts)")
    print("="*70)

    print("\ny_t.l (Terminal Visits):")
    for i in model.i:
        visits = [(k, safe_value(model.y_t[i,k])) for k in model.k
                 if safe_value(model.y_t[i,k]) > 0]
        if visits:
            total_terminal_visits = sum(v for _, v in visits)
            print(f"  {i} (Total: {total_terminal_visits:.0f} visits):")
            for k, v in visits:
                print(f"    {k}: {v:.0f} visits")

    print("\ny_p.l (Power Plant Visits):")
    for j in model.j:
        visits = [(k, safe_value(model.y_p[j,k])) for k in model.k
                 if safe_value(model.y_p[j,k]) > 0]
        if visits:
            total_visits = sum(v for _, v in visits)
            print(f"  {j} (Total: {total_visits:.0f} visits):")
            for k, v in visits:
                print(f"    {k}: {v:.0f} visits")

    # ========== Display: Continuous Variables ==========
    print("\n" + "="*70)
    print("DISPLAY DIST.l, Time.l, Tmr.l, M.l (Continuous Variables)")
    print("="*70)

    print(f"\n{'Ship':<15} {'DIST(km)':<15} {'Time(h)':<15} {'Tmr(h)':<15} {'M(m³)':<15} {'Util(%)':<10}")
    print("-" * 85)

    for k in active_ships:
        dist = pyomo.value(model.DIST[k]) if model.DIST[k].value else 0
        time = pyomo.value(model.Time[k]) if model.Time[k].value else 0
        tmr = pyomo.value(model.Tmr[k]) if model.Tmr[k].value else 0
        m = pyomo.value(model.M[k]) if model.M[k].value else 0
        util = (m / model.capacity_ship[k] * 100) if model.capacity_ship[k] > 0 else 0

        print(f"{k:<15} {dist:<15,.2f} {time:<15,.2f} {tmr:<15,.2f} {m:<15,.2f} {util:<10.1f}")

    print(f"\nTotal Active Ships: {len(active_ships)} / {len(model.k)}")

    # Ship type breakdown
    shinju_count = len([k for k in active_ships if 'Shinju' in k])
    wsd59_count = len([k for k in active_ships if 'WSD59' in k])
    coral_count = len([k for k in active_ships if 'Coral' in k])

    print(f"\nShips by Type:")
    print(f"  Shinju (2,513 m³): {shinju_count}")
    print(f"  WSD59 (5,000 m³): {wsd59_count}")
    print(f"  Coral (7,500 m³): {coral_count}")

    # ========== Display: TANK Variables ==========
    print("\n" + "="*70)
    print("DISPLAY TANK.l (LNG Storage at Power Plants)")
    print("="*70)

    print(f"\n{'Plant':<15} {'TANK(m³)':<15} {'Demand(m³)':<15} {'Satisfaction(%)':<15}")
    print("-" * 60)

    for j in model.j:
        tank = pyomo.value(model.TANK[j]) if model.TANK[j].value else 0
        demand_required = model.delivery_period * model.demand[j]
        tank_capacity = model.delivery_period * model.capacity_pp[j]
        satisfaction = (tank / demand_required * 100) if demand_required > 0 else 0
        status = "OK" if tank >= demand_required * 0.99 else "X"

        print(f"{j:<15} {tank:<15,.2f} {demand_required:<15,.2f} {satisfaction:<10.1f}% {status}")

    # ========== Display: Slack Variables ==========
    print("\n" + "="*70)
    print("DISPLAY slack_meq.l, slack_time.l (Slack Variables)")
    print("="*70)

    total_slack_meq = sum(pyomo.value(model.slack_meq[k]) for k in model.k)
    total_slack_time = sum(pyomo.value(model.slack_time[k]) for k in model.k)

    print(f"\nTotal Capacity Slack: {total_slack_meq:.4f} m³")
    print(f"Total Time Slack: {total_slack_time:.4f} hours")

    if total_slack_meq > 0.01 or total_slack_time > 0.01:
        print("\nShips with non-zero slack:")
        print(f"{'Ship':<15} {'Capacity Slack(m³)':<20} {'Time Slack(h)':<20}")
        print("-" * 55)
        for k in model.k:
            slack_m = pyomo.value(model.slack_meq[k])
            slack_t = pyomo.value(model.slack_time[k])
            if slack_m > 0.01 or slack_t > 0.01:
                print(f"{k:<15} {slack_m:<20,.4f} {slack_t:<20,.4f}")
    else:
        print("All constraints satisfied without slack variables")

    # ========== Route Summary ==========
    print("\n" + "="*70)
    print("ROUTE SUMMARY (Complete Delivery Paths)")
    print("="*70)

    # Sort ships by capacity for organized display
    active_ships_sorted = sorted(active_ships, key=lambda k: model.capacity_ship[k], reverse=True)

    # Delivery type statistics
    milkrun_ships = []      # Milk-Run
    direct_ships = []       # Direct delivery 
    overloaded_ships = []   # Overloaded ships

    for k in active_ships_sorted: 
        capacity = model.capacity_ship[k]
        volume = pyomo.value(model.M[k])
        utilization = (volume / capacity) * 100

        print(f"\n{k} (Capacity: {capacity:.0f} m³, Utilization: {utilization:.1f}%):")

        # Build complete route 
        route = []
        
        # Find starting terminal and first plant
        starting_terminal = None
        first_plant = None
        for i in model.i:
            for j in model.j:
                if safe_value(model.x_t_p[i,j,k]) > 0.5:
                    starting_terminal = i
                    first_plant = j
                    route.append(i)
                    route.append(j)
                    break
            if starting_terminal:
                break
        
        # Follow plant-to-plant connections (Milk-Run Path)
        if starting_terminal and first_plant:
            current_plant = first_plant
            visited = set([current_plant])
            max_iterations = len(model.j)
            iteration = 0
            
            while iteration < max_iterations:
                found_next = False
                for j2 in model.j:
                    if j2 not in visited and safe_value(model.x_p_p[current_plant,j2,k]) > 0.5:
                        route.append(j2)
                        visited.add(j2)
                        current_plant = j2
                        found_next = True
                        break
                
                if not found_next:
                    break
                iteration += 1
            
            # Find return terminal
            for i in model.i:
                if safe_value(model.x_p_t[current_plant,i,k]) > 0.5:
                    route.append(i)
                    break
            
            # Display route
            if len(route) >= 3:
                route_str = " → ".join(route)
                print(f"  Route: {route_str}")
                
                # Count plants visited
                plants_visited = len([x for x in route if x in model.j])
                
                # Determine delivery type based on 3-Tier Priority System
                allow_single_k = pyomo.value(model.allow_single_visit[k]) > 0.5 if model.allow_single_visit[k].value is not None else False
                
                ship_info = {
                    'ship': k, 'plants': plants_visited,
                    'volume': volume, 'distance': pyomo.value(model.DIST[k]),
                    'utilization': utilization
                }

                # Classify based on 3-Tier Priority System
                # Check slack values
                slack_meq_val = pyomo.value(model.slack_meq[k]) if model.slack_meq[k].value is not None else 0
                slack_time_val = pyomo.value(model.slack_time[k]) if model.slack_time[k].value is not None else 0

                if allow_single_k and plants_visited == 1:
                    # PRINCIPLE 3: Route split allowed
                    print(f"  Delivery Type: PRINCIPLE 3 - Route Split - {plants_visited} plant [Util >= 100%]")
                    overloaded_ships.append(ship_info)
                elif plants_visited >= 2:
                    # PRINCIPLE 1: Milk-Run (2+ plants) - Base strategy
                    # May use PRINCIPLE 2 (larger ship upgrade) if needed
                    ship_type = 'Shinju' if 'Shinju' in k else ('WSD59' if 'WSD59' in k else 'Coral')
                    print(f"  Delivery Type: PRINCIPLE 1 - Milk-Run ({plants_visited} plants) [{ship_type}, Util: {utilization:.1f}%]")
                    milkrun_ships.append(ship_info)
                elif plants_visited == 1 and not allow_single_k:
                    # Edge case: Single plant but not flagged for split
                    print(f"  Delivery Type: Direct ({plants_visited} plant) [Low Util: {utilization:.1f}%]")
                    direct_ships.append(ship_info)
                else:
                    # Fallback
                    print(f"  Delivery Type: Unknown ({plants_visited} plants) [Util: {utilization:.1f}%]")
                    direct_ships.append(ship_info)
                
                # Display slack values
                slack_status = "OK" if (slack_meq_val < 0.01 and slack_time_val < 0.01) else "WARNING"
                print(f"  Slack: Capacity={slack_meq_val:.2f}, Time={slack_time_val:.2f} [{slack_status}]")

                # Ship performance details
                print(f"  Volume Delivered: {pyomo.value(model.M[k]):,.0f} m³")
                print(f"  Distance Traveled: {pyomo.value(model.DIST[k]):,.0f} km")
                print(f"  Total Time: {pyomo.value(model.Time[k]):,.1f} hours")
                print(f"  Sea Time: {pyomo.value(model.Tmr[k]):,.1f} hours")    
            else:
                # Route is too short (less than 3 nodes)
                print("  Route: Incomplete route detected")
                print(f"  Volume Delivered: {pyomo.value(model.M[k]):,.0f} m³")
                print(f"  Distance Traveled: {pyomo.value(model.DIST[k]):,.0f} km")    
        else:
            # No starting terminal or first plant found
            print("  Route: Could not determine route (no terminal-to-plant connection)")
        
            # Ship performance details
            print(f"  Volume Delivered: {pyomo.value(model.M[k]):,.0f} m³")
            print(f"  Distance Traveled: {pyomo.value(model.DIST[k]):,.0f} km")
            print(f"  Total Time: {pyomo.value(model.Time[k]):,.1f} hours")
            print(f"  Sea Time: {pyomo.value(model.Tmr[k]):,.1f} hours")    

    # ========== Summary Statistics ==========
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    total_distance = sum(pyomo.value(model.DIST[k]) for k in model.k if model.DIST[k].value)
    total_volume = sum(pyomo.value(model.M[k]) for k in model.k if model.M[k].value)
    total_demand = sum(model.demand[j] * model.delivery_period for j in model.j)

    print(f"\nOperational Metrics:")
    print(f"  Total Ships Used: {len(active_ships)} / {len(model.k)}")
    print(f"  Total Distance: {total_distance:,.0f} km")
    print(f"  Total Volume Delivered: {total_volume:,.0f} m³")
    print(f"  Total Demand Required: {total_demand:,.0f} m³")
    print(f"  Demand Satisfaction Rate: {(total_volume/total_demand)*100:.2f}%")

    print(f"\nCost Summary:")
    print(f"  Total Operating Cost (OPEX): ${pyomo.value(model.OPEX):,.2f}")
    print(f"  Fuel Cost: ${total_fuel:,.2f}")
    print(f"  Ship Rental: ${total_ship:,.2f}")
    print(f"  Port Fees (Tug+Harbor+Guide): ${(total_tug+total_harbor+total_guide):,.2f}")

    print(f"\nSolution Quality:")
    print(f"  Computation Time: {solve_time:.2f} seconds")
    
      # ========== Capacity-Based Constraint Verification ==========
    print("\n" + "="*70)
    print("CAPACITY-BASED CONSTRAINT VERIFICATION")
    print("="*70)

    # Get ship lists
    shinju_ships = [k for k in model.k if k.startswith('Shinju')]
    wsd59_ships = [k for k in model.k if k.startswith('WSD59')]
    coral_ships = [k for k in model.k if k.startswith('Coral')]

    # Calculate total capacities
    shinju_capacity = sum(model.ship_used[k].value * 2513
                          for k in shinju_ships if model.ship_used[k].value > 0.5)
    wsd59_capacity = sum(model.ship_used[k].value * 5000
                         for k in wsd59_ships if model.ship_used[k].value > 0.5)
    coral_capacity = sum(model.ship_used[k].value * 7500
                         for k in coral_ships if model.ship_used[k].value > 0.5)

    # Count ships used
    shinju_count = sum(1 for k in shinju_ships if model.ship_used[k].value > 0.5)
    wsd59_count = sum(1 for k in wsd59_ships if model.ship_used[k].value > 0.5)
    coral_count = sum(1 for k in coral_ships if model.ship_used[k].value > 0.5)
    
    print(f"\nShip Usage Summary:")
    print(f"  Shinju: {shinju_count} ships → {shinju_capacity:,.0f} m³ total capacity")
    print(f"  WSD59:  {wsd59_count} ships → {wsd59_capacity:,.0f} m³ total capacity")
    print(f"  Coral:  {coral_count} ships → {coral_capacity:,.0f} m³ total capacity")

    # Calculate total demand for buffer calculation
    total_demand = sum(model.delivery_period * model.demand[j] for j in model.j)

    # Calculate adaptive buffers (same logic as in constraints)
    # ADAPTIVE ENHANCED: Period-specific coefficients (incraesed for strict 100% utilization)
    if model.delivery_period <= 7:
        wsd59_coeff = 0.61 # Gap: 0.09
        coral_coeff = 0.50
    elif model.delivery_period <= 10:
        wsd59_coeff = 0.60 # Gap: 0.10
        coral_coeff = 0.50
    elif model.delivery_period <= 15:
        wsd59_coeff = 0.60 # Gap: 0.10
        coral_coeff = 0.50
    else: # >= 20
        wsd59_coeff = 0.59 # Gap: 0.09
        coral_coeff = 0.50
        wsd59_buffer = total_demand * wsd59_coeff

    wsd59_buffer = total_demand * wsd59_coeff
    coral_buffer = total_demand * coral_coeff

    # WSD59 constraint check
    wsd59_limit = shinju_capacity * 1.5 + wsd59_buffer
    wsd59_satisfied = wsd59_capacity <= wsd59_limit

    print(f"\nWSD59 Priority Constraint (ADAPTIVE DYNAMIC BUFFER):")
    print(f"  Formula: wsd59_capacity <= shinju_capacity * 1.5 + (total_demand * {wsd59_coeff:.2f})")
    print(f"  Buffer: {wsd59_buffer:,.0f} m³ ({wsd59_coeff*100:.0f}% of {total_demand:,.0f} m³)")
    print(f"  Calculation: {wsd59_capacity:,.0f} <= {shinju_capacity:,.0f} * 1.5 + {wsd59_buffer:,.0f}")
    print(f"  Result: {wsd59_capacity:,.0f} <= {wsd59_limit:,.0f}")
    print(f"  Status: {'SATISFIED' if wsd59_satisfied else 'VIOLATED'}")
    if not wsd59_satisfied:
        print(f"  WARNING: Constraint violated by {wsd59_capacity - wsd59_limit:,.0f} m³")

    # Coral constraint check
    coral_limit = wsd59_capacity * 1.2 + coral_buffer
    coral_satisfied = coral_capacity <= coral_limit

    print(f"\nCoral Priority Constraint (DYNAMIC BUFFER):")
    print(f"  Formula: coral_capacity <= wsd59_capacity * 1.2 + (total_demand * {coral_coeff:.2f})")
    print(f"  Buffer: {coral_buffer:,.0f} m³ ({coral_coeff*100:.0f}% of {total_demand:,.0f} m³)")
    print(f"  Calculation: {coral_capacity:,.0f} <= {wsd59_capacity:,.0f} * 1.2 + {coral_buffer:,.0f}")
    print(f"  Result: {coral_capacity:,.0f} <= {coral_limit:,.0f}")
    print(f"  Status: {'SATISFIED' if coral_satisfied else 'VIOLATED'}")
    if not coral_satisfied:
        print(f"  WARNING: Constraint violated by {coral_capacity - coral_limit:,.0f} m³")

    # Overall status
    all_satisfied = wsd59_satisfied and coral_satisfied
    print(f"\nOverall Constraint Status: {'ALL CONSTRAINTS SATISFIED' if all_satisfied else 'CONSTRAINT VIOLATIONS DETECTED'}")
    
    # ========== Delivery Type Analysis: Utilization-Based ==========
    print("\n" + "="*70)
    print("UTILIZATION-BASED DELIVERY ANALYSIS")
    print("="*70)

    # Overloaded ships (Utilization > 100%)
    if overloaded_ships:
        print(f"\nOverloaded Ships - Route Split ({len(overloaded_ships)} ships):")
        print(f"{'Ship':<15} {'Plants':<8} {'Util(%)':<10} {'Volume(m³)':<12} {'Distance(km)':<12}")
        print("-" * 65)
        total_ol_volume = 0
        total_ol_distance = 0
        for ship in overloaded_ships:
            print(f"{ship['ship']:<15} {ship['plants']:<8} {ship['utilization']:<10.1f} {ship['volume']:<12,.0f} ")
            total_ol_volume += ship['volume']
            total_ol_distance += ship['distance']
        print("-" * 65)
        print(f"{'Total':<15} {'':<8} {'':<10} {total_ol_volume:<12,.0f} "
              f"{total_ol_distance:<12,.0f}")
        
    # Milk-Run ships (50% < Utilization <= 100%)
    if milkrun_ships:
        print(f"\nMilk-Run Ships ({len(milkrun_ships)} ships):")
        print(f"{'Ship':<15} {'Plants':<8} {'Util(%)':<10} {'Volume(m³)':<12} {'Distance(km)':<12}")
        print("-" * 65)    
        total_mr_volume = 0
        total_mr_distance = 0
        for ship in milkrun_ships:
            print(f"{ship['ship']:<15} {ship['plants']:<8} {ship['utilization']:<10.1f} {ship['volume']:<12,.0f} "
                f"{ship['distance']:<12,.0f}")
            total_mr_volume += ship['volume']
            total_mr_distance += ship['distance']
        print("-" * 65)
        print(f"{'Total':<15} {'':<8} {'':<10} {total_mr_volume:<12,.0f} "
              f"{total_mr_distance:<12,.0f}")

    # Direct ships (Utilization <= 50%)
    if direct_ships:
        print(f"\nDirect Ships (Util <= 50%) ({len(direct_ships)} ships):")
        print(f"{'Ship':<15} {'Plants':<8} {'Util(%)':<10} {'Volume(m³)':<12} {'Distance(km)':<12}")
        print("-" * 65)
        total_dr_volume = 0
        total_dr_distance = 0
        for ship in direct_ships:
            print(f"{ship['ship']:<15} {ship['plants']:<8} {ship['utilization']:<10.1f} {ship['volume']:<12,.0f} "
                  f"{ship['distance']:<12,.0f}")
            total_dr_volume += ship['volume']
            total_dr_distance += ship['distance']
        print("-" * 65)
        print(f"{'Total':<15} {'':<8} {'':<10} {total_dr_volume:<12,.2f} "
              f"{total_dr_distance:<12,.0f}")

    # Summary statistics
    total_all = len(overloaded_ships) + len(milkrun_ships) + len(direct_ships)
    print(f"\nDelivery Type Summary:")
    print(f"  - Route Split: {len(overloaded_ships)} ships ({len(overloaded_ships)/total_all*100:.1f}%)" if total_all > 0 else "")
    print(f"  - Milk-Run (2+ plants): {len(milkrun_ships)} ships ({len(milkrun_ships)/total_all*100:.1f}%)" if total_all > 0 else "")
    print(f"  - Direct (1 plant): {len(direct_ships)} ships ({len(direct_ships)/total_all*100:.1f}%)" if total_all > 0 else "")
else:
    print(f"\n⚠ Optimization terminated with status: {results.solver.termination_condition}")
    print("NO optimal solution found.")

