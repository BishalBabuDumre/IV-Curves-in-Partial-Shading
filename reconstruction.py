#Imports
import pandas as pd
import math
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit

#Creating Bold Graph Edges and Fonts
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 16

#Function to clean input CSV file
def cleanData(input_file, index):
    df = pd.read_csv(input_file)
    x = df["volts_curve"].tolist()[index]
    y = df["amps_curve"].tolist()[index]
    x = list(map(float, x[1:-1].split(", ")))
    y = list(map(float, y[1:-1].split(", ")))
    return x, y

#Function to extract forward single and (n-1) cell data
def forwardPart(voltData,currData):
    voltSS = [j / 96 for j in voltData]  # Getting voltage for single cell
    voltBB = [p*95 for p in voltSS] #Calculating voltage for unshaded cells
    return voltSS, voltBB

#Extracting the unshaded part from CSV file1
voltF1, currF1 = cleanData("2024-11-04.csv",69)
voltFSC1, voltFF1 = forwardPart(voltF1, currF1)

#Extracting the shaded part from CSV file1
voltS1, currS1 = cleanData("2024-11-04.csv",68)

#Extracting the unshaded part from CSV file2
voltF2, currF2 = cleanData("2024-11-04.csv",69)
voltFSC2, voltFF2 = forwardPart(voltF2, currF2)

#Extracting the shaded part from CSV file2
voltS2, currS2 = cleanData("2024-11-04.csv",70)

#Extracting the unshaded part from CSV file2
voltF3, currF3 = cleanData("2024-11-04.csv",71)
voltFSC3, voltFF3 = forwardPart(voltF3, currF3)

#Extracting the shaded part from CSV file2
voltS3, currS3 = cleanData("2024-11-04.csv",72)

#Extracting the unshaded part from CSV file2
voltF4, currF4 = cleanData("2024-11-04.csv",73)
voltFSC4, voltFF4 = forwardPart(voltF4, currF4)

#Extracting the shaded part from CSV file2
voltS4, currS4 = cleanData("2024-11-04.csv",74)

#Function to interpolate forward 95 cells to subtract from full shaded module
def diffVolt(voltShade, currShade, voltBright, currBright):
    fx = interpolate.interp1d(currBright, voltBright, bounds_error=False, fill_value="extrapolate")
    voltBF = fx(currShade)
    voltSub = [u-v for u, v in zip(voltShade, voltBF)] #Subtracting to obtain reverse character
    return voltSub

voltDiff1 = diffVolt(voltS1, currS1, voltFF1, currF1)
voltDiff2 = diffVolt(voltS2, currS2, voltFF2, currF2)
voltDiff3 = diffVolt(voltS3, currS3, voltFF3, currF3)
voltDiff4 = diffVolt(voltS4, currS4, voltFF4, currF4)

#Clipping the reverse characteristics
def clipReverse(voltFarak, curr):
    voltFarak, curr = zip(*[(a, b) for a, b in zip(voltFarak, curr) if not math.isinf(a)])
    if min(voltFarak) < -5:
        limit = -5
    else:
        limit = min(voltFarak)
        voltFarak, curr = zip(*[(a, b) for i, (a, b) in enumerate(zip(voltFarak, curr)) if i >= voltFarak.index(limit)])
    voltRe = np.linspace(limit, 0, 100) #Clipping from -5V to 0V
    interp_function = interpolate.interp1d(voltFarak, curr, bounds_error=False, fill_value="extrapolate")
    currRe = interp_function(voltRe)
    currRe = [m-currRe[len(currRe)-1] for m in currRe] #Shifting the curve to the origin
    return voltRe, currRe

voltR1, currR1 = clipReverse(voltDiff1, currS1)
voltR2, currR2 = clipReverse(voltDiff2, currS2)
voltR3, currR3 = clipReverse(voltDiff3, currS3)
voltR4, currR4 = clipReverse(voltDiff4, currS4)

#Extrapolating the reverse character at the point of max dy/dx
def exponential_model(x, a, b):
    return a * np.exp(b * x)

#Function to extrapolate reverse curve
def extraPolate(voltP, currP, smoothing_factor=0.1):
    #Determining index of max dy/dx to extrapolate reverse curve
    i = -1
    dy_dx = np.gradient(currP, voltP)  # First derivative
    min_value = np.min(dy_dx)
    i = np.argmin(dy_dx)

    #Extrapolating the reverse character at the point of max dy/dx
    # Fit the model to the data
    params, covariance = curve_fit(exponential_model, voltP[i:i+7], currP[i:i+7])
    a, b = params  # Fitted parameters

    # Extrapolation: Define new X values
    x_Ex = np.linspace(-4.5, min(voltP[i:]), 100)  # Extend range for extrapolation
    y_Ex = exponential_model(x_Ex, a, b)

    x_final = np.append(x_Ex, voltP[i:])
    y_final = np.append(y_Ex, currP[i:])
    return x_final, y_final

voltEx1, currEx1 = extraPolate(voltR1, currR1)
voltEx2, currEx2 = extraPolate(voltR2, currR2)
voltEx3, currEx3 = extraPolate(voltR3, currR3)
voltEx4, currEx4 = extraPolate(voltR4, currR4)


#Final Modeling by adding shaded and unshaded portions
def finalAdd(currA, voltK, voltJ, voltC, currC, shadePercent):
    currFP = [k*(100-shadePercent)/100 for k in currA] # Assuming 95% shading
    print("Current at Inflection Point: ",currFP[0])
    currRR = [q+currFP[0] for q in currC] #Shifting the current of reverse bias at the Isc of shaded cell
    currRRR = np.append(currRR, currFP) #Extending the reverse bias current with forward bias current to create overall current
    voltFFF = np.append(voltC, voltK) #Extending the reverse bias voltage with forward bias voltage to create overall voltage
    #Interpolating the reverse shaded part to add with forward 95 unshaded ones.
    fx1 = interpolate.interp1d(currRRR, voltFFF, bounds_error=False, fill_value="extrapolate")
    currcurr = np.linspace(min(currA), max(currA), 1000) #Clipping from -5V to 0V
    voltage = fx1(np.array(currcurr))
    fx2 = interpolate.interp1d(currA, voltJ, bounds_error=False, fill_value="extrapolate")
    voltJJ = fx2(np.array(currcurr))
    finalVolt = [u+v for u, v in zip(voltJJ, voltage)]
    print("Voltage at Inflection Point: ", fx1(currFP[0])+fx2(currFP[0]))
    return finalVolt, currcurr

finalBij1, finalcurr1 = finalAdd(currF1, voltFSC1, voltFF1, voltEx1, currEx1, 93)
finalBij2, finalcurr2 = finalAdd(currF1, voltFSC1, voltFF1, voltEx1, currEx1, 70)
finalBij3, finalcurr3 = finalAdd(currF2, voltFSC2, voltFF2, voltEx1, currEx1, 57)
finalBij4, finalcurr4 = finalAdd(currF4, voltFSC4, voltFF4, voltEx1, currEx1, 28)

# Compute the products
products = [v * c for v, c in zip(voltS1, currS1)]

# Find the maximum product and its index
max_product = max(products)
max_index = products.index(max_product)

# Retrieve the values from the original lists
voltage_at_max = voltS1[max_index]
current_at_max = currS1[max_index]

#Final Plotting
plt.plot(voltS1, currS1, c = 'b', ls = "-", lw = 1.5, label = "Original-1 (93% Shading)")
plt.plot(finalBij1, finalcurr1, c = 'orange', ls = ":", lw = 1.5, label = "Modeled-1 (93% Shading)")
plt.plot(voltS2, currS2, c = 'r', ls = "--", lw = 1.5, label = "Original-2 (70% Shading)")
plt.plot(finalBij2, finalcurr2, c = 'cyan', ls = "-.", lw = 1.5, label = "Modeled-2 (70% Shading)")
plt.plot(voltS3, currS3, c = 'g', ls = (0, (5, 5)), lw = 1.5, label = "Original-3 (57% Shading)")
plt.plot(finalBij3, finalcurr3, c = '#FFFF00', ls = (0, (1, 1)), lw = 1.5, label = "Modeled-3 (57% Shading)")
plt.plot(voltS4, currS4, c = 'teal', ls = (0, (10, 5, 3, 5)), lw = 1.5, label = "Original-4 (28% Shading)")
plt.plot(finalBij4, finalcurr4, c = 'purple', ls = (0, (3, 1, 1, 1)), lw = 1.5, label = "Modeled-4 (28% Shading)")
plt.plot(63.93702591168992, 0.40317844, c = 'k', marker = "x", ms = 5)
plt.plot(62.70848687777844, 1.7279076, c = 'k', marker = "x", ms = 5)
plt.plot(61.87717188773692, 2.47666756, c = 'k', marker = "x", ms = 5)
plt.plot(59.59024784154093, 4.13873568, c = 'k', marker = "x", ms = 5)
plt.text(18.5, 1, '= Inflection Point', fontsize = 10)
plt.plot(17, 1.08, c = 'k', marker = "x", ms = 5)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
plt.title("RECONSTRUCTION OF IV-CURVES")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.legend(bbox_to_anchor = (0.45, 0.85))
plt.savefig("Graph4.pdf")
plt.close()
