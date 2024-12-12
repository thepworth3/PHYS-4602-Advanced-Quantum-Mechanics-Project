# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:46:29 2024

@author: thoma
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Constants
a = 1
hbar = 1
mass = 1
Num = 10  # Number of basis states

# Potential function
def V(x):
    return 100 * x/a

def Vquad(x):
    return 100*x**2/a**2

# Wavefunction
def Psi(n, x):
    return np.sqrt(2 / a) * np.sin(n * np.pi * x / a)

# Trapezoidal integrator to make it easier to call
def integrate_trapz(f, x):
    return np.trapz(f, x)


# Compute the off-diagonal elements fromt the V expectation value 
def VO(n, m, x_vals):
    integrand = Psi(n, x_vals) * V(x_vals) * Psi(m, x_vals)
    return integrate_trapz(integrand, x_vals)

# Compute the on-diagonal elements
def VD(n, x_vals):
    integrand = Psi(n, x_vals) * V(x_vals) * Psi(n, x_vals)
    return integrate_trapz(integrand, x_vals)

# Compute the off-diagonal elements fromt the V expectation value 
def VquadO(n, m, x_vals):
    integrand = Psi(n, x_vals) * Vquad(x_vals) * Psi(m, x_vals)
    return integrate_trapz(integrand, x_vals)

# Compute the on-diagonal elements
def VquadD(n, x_vals):
    integrand = Psi(n, x_vals) * Vquad(x_vals) * Psi(n, x_vals)
    return integrate_trapz(integrand, x_vals)


# Create array of x values between 0 and a for integration
x_vals = np.linspace(0, a, 1000)

# Build Hamiltonian matrix, 10 x 10 in this case
H = np.zeros((Num, Num))


for n in range(1, Num + 1):
    for m in range(1, Num + 1):
        if n == m:
            H[n - 1, m - 1] = (n ** 2 * np.pi **2 * hbar**2) / (2*mass * a**2) + VD(n, x_vals)
        else:
            H[n - 1, m - 1] = VO(n, m, x_vals)

# print(H)
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(H)
# print(eigenvalues)
E0 = min(eigenvalues)  # Highest eigenvalue

# Normalize the eigenvectors
cl = eigenvectors[:, 0]

cl /= np.linalg.norm(cl) #normalize the c vector 



# Compute ψ(x)
def psil(x):
    return sum(cl[i] * Psi(i + 1, x) for i in range(Num))   #am i not multiplying through properly?

# we will plot both wavefunctions together later

# print(np.trapz(psi_vals, x_vals))

# Print the lowest eigenvalue
print("E0 =", E0)
print("Difference from exact solution is", (E0 - 39.9819)/39.9819*100,"%")

#Now repeating for quadratic potential
for n in range(1, Num + 1):
    for m in range(1, Num + 1):
        if n == m:
            H[n - 1, m - 1] = (n ** 2 * np.pi **2 * hbar**2) / (2*mass * a**2) + VquadD(n, x_vals)
        else:
            H[n - 1, m - 1] = VquadO(n, m, x_vals)

# print(H)
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(H)
E0 = min(eigenvalues)  # Highest eigenvalue
ev = eigenvectors[0]
# Normalize the eigenvectors
c = eigenvectors[:, 0]

c /= np.linalg.norm(c)
print(c)



# Compute ψ(x)
def psi(x):
    return sum(c[i] * Psi(i + 1, x) for i in range(Num))   

# Plot ψ(x)
plt. figure(dpi=100)

x_plot = np.linspace(0, a, 1000)
psi_vals = np.array([psi(x) for x in x_plot])
plt.plot(x_plot, psi_vals, label="Quadratic Potential")

psi_valsl = np.array([psil(x) for x in x_plot])
plt.plot(x_plot, psi_valsl, label="Linear Potential")

plt.xlabel('x')
plt.ylabel(r'$\psi(x)$')
plt.title('Optimal Wavefunction')
x_ticks = np.arange(0, a + 0.25, 0.25)  # From 0 to a with step 0.25
y_ticks = np.arange(0, 2.0, 0.25)     # Adjust range for psi(x)

plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.legend()

plt.show()

print("E0 =", E0)

#WKB Stuff
n = 4
a = 1
hbar = 1
mass = 1

#solve for the roots 
from scipy import optimize 


#the qunaitzation integral so that we can find the energy roots
#note V(a) = V_o
def energy(E):
    return np.sqrt(2*mass)*2*a/(3*V(a))*(E**(3/2) - (E - V(a))**(3/2)) - n*np.pi*hbar

sol = optimize.root_scalar(energy, x0 = 101)


E = sol.root
print("root", sol.root)


print(E)

def p(x):
    return np.sqrt(((2*mass*(E-V(x)))))

# n = 1
# print(p(x_vals))

def phi(x):
    return -2*np.sqrt(2*mass)*a/(3*V(a)*hbar)*( (E - V(x))**(3/2) - E**(3/2)  )


def C(x):
    return np.sqrt(1/np.trapz((np.sin(phi(x))*np.sin(phi(x)))/p(x), x))

print("The norm constant is C = ", C(x_vals))

def psiwkb(x):
    return C(x)*np.sin(phi(x))*(1/np.sqrt((p(x))))

psi = psiwkb(x_vals)

print("norm", np.trapz(psi**2, x_vals))
plt. figure(dpi=100)
plt.plot(x_vals, psi, label = "WKB")
plt.xlabel('x')
plt.ylabel(r'$\psi_4(x)$')
plt.title('WKB Approximate Wavefunction')
x_ticks = np.arange(0, a + 0.25, 0.25)  # From 0 to a with step 0.25
y_ticks = np.arange(0, 2.0, 0.25)     # Adjust range for psi(x)
plt.legend()
# plt.plot(x_vals, phi(x_vals))

# plt.plot(x_vals, V(x_vals), label = "WKB")
# plt.plot(x_vals,np.full(len(x_vals),E))
plt.show()




### WKB for the quadratic potential

def pquad(x):
    return np.sqrt(((2*mass*(E-Vquad(x)))))


n = 5
a = 1
hbar = 1
mass = 1

def quadenergy(E):
    return (a*E*np.sqrt(mass)/(np.sqrt(2*Vquad(a))))*( np.arcsin(np.sqrt(Vquad(a)/E)) + 0.5*np.sin(2*np.arcsin(np.sqrt(Vquad(a)/E)))) - n*np.pi*hbar


solquad = optimize.root_scalar(quadenergy, x0 = 101)

E = solquad.root
print("energy", E)

def phiquad(x):
    return a*E*np.sqrt(mass)*(np.arcsin((x*np.sqrt(Vquad(a))/(a*np.sqrt(E)))) + 0.5*np.sin(2*np.arcsin(x*np.sqrt(Vquad(a))/(a*np.sqrt(E)))   ))/(np.sqrt(2*Vquad(a))*hbar)


def Cquad(x):
    return np.sqrt(1/np.trapz((np.sin(phiquad(x))*np.sin(phiquad(x)))/pquad(x), x))

print("The norm constant is C = ", C(x_vals))

def psiwkbquad(x):
    return Cquad(x)*np.sin(phiquad(x))*(1/np.sqrt((pquad(x))))

plt.figure(dpi = 1200)
plt.plot(x_vals, psiwkbquad(x_vals), label = "Quadratic")
plt.xlabel('x')
plt.ylabel(r'$\psi_5(x)$')
plt.title('WKB Approximate Wavefunction')
plt.legend()
plt.show()

print("norm", np.trapz(psiwkbquad(x_vals)**2, x_vals))
