import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy import units as u
from astropy.units import cds
from astropy.constants import sigma_sb, G
from astropy.timeseries import LombScargle

plt.style.use('ggplot')

data = pd.read_csv('data.txt', skiprows=10)
timee = data.time.to_numpy() #[MJD=59731+]
rvv = data.rv.to_numpy()/1000 #[km/s]

time = timee*cds.MJD
rv = rvv*u.km/u.s

s_mass = 1*u.Msun
td = 0.0000837 #transit depth
tl = 13.1*u.hr #transit lenght

frequency, power = LombScargle(timee, rvv).autopower()
best_frequency = frequency[np.argmax(power)]
period = 1/best_frequency
'''
#gráfico de frecuencias
plt.plot(frequency, power, color='#444444')
plt.xlabel('Orbital Frequency [1/days]')
plt.ylabel('Power') #power is adimensionless
plt.savefig('frequency.png', dpi=300)
plt.show()
'''
#función para el one phase
time_ = []
for i in timee:
    time_.append(i%period)
timee = time_ #redefiniendo la variable, ahora solo con los datos en solo una fase

def model(t, A, B, C):
    return (A*np.cos(2*np.pi*best_frequency*t+B)+C)
fit= curve_fit(model, timee, rvv)
ans, cov= fit
fit_a, fit_b, fit_c = ans
t= np.linspace(min(timee), max(timee), 200)
plt.scatter(timee, rvv, s=15, color='#444444', label='data')
plt.plot(t, model(t, fit_a, fit_b, fit_c), color='#4DADC0', label='model')

plt.ylabel('Radial Velocity [km s$^{-1}$]')
plt.legend(loc='best')
plt.savefig('model.png', dpi=300)
plt.show()

#error
rv_error=[]
k=0
while k < 100:
    rv_error.append(rvv[k]-model(timee[k],fit_a,fit_b,fit_c) )
    k+=1
plt.scatter(timee, rv_error, s=15, color='#444444')
plt.xlabel('Time [days]')
plt.ylabel('Error [km s$^{-1}$]')
plt.savefig('error.png', dpi=300)
plt.show()



#datos de la estrella y el planeta
v_obs = fit_a*u.km/u.s #semiamplitud de la velocidad radial en función del tiempo
period = 1/best_frequency*cds.MJD
s_luminosity = 1*cds.Lsun*(s_mass/(1*u.Msun))**4
orbital_separation = (((period**2)*G*s_mass/(4*(np.pi**2)))**(1/3)).decompose().to(u.au)
s_radius = (tl*np.pi*orbital_separation/period).decompose().to(u.Rsun)
s_temperature = ((s_luminosity/(4*np.pi*s_radius**2*sigma_sb))**(1/4)).si
p_radius = ((td*(s_radius**2))**(1/2)).decompose().to(u.Rearth)
l = orbital_separation*(np.sin((np.pi*tl/period)*u.rad)).decompose()
b = (((s_radius+p_radius)**2-l**2)**(1/2))/s_radius
i = (np.arccos(b*s_radius/orbital_separation)).decompose().to(u.deg)
p_mass = (((s_mass**2*period*v_obs**3)/(2*np.pi*G))**(1/3)).decompose().to(u.Mearth)*np.sin(i)
p_density = (p_mass/((4/3)*np.pi*p_radius**3)).decompose().to(u.kg/(u.m**3))
p_eqt = (s_temperature*((1-0.3)**(1/4))*((s_radius/(2*orbital_separation))**(1/2))).decompose().to(u.K)

#archivo con datos obtenidos de la estrella y el planeta
file = open('star-planet_data.txt', 'w')
file.write('Inclination of the system: ' + str(i) + '\n\n')
file.write('Planet \n')
file.write('Mass*sin(i): ' + str(p_mass) + '\n')
file.write('Radius: ' + str(p_radius) + '\n')
file.write('Density: ' + str(p_density) + '\n')
file.write('Period: ' + str(period) + '\n')
file.write('Orbital separation: ' + str(orbital_separation) + '\n')
file.write('Equilibrium temperature: ' + str(p_eqt) + '\n\n')
file.write('Star \n')
file.write('Mass: ' + str(s_mass) + '\n')
file.write('Radius: ' + str(s_radius) + '\n')
file.write('Luminosity: ' + str(s_luminosity) + '\n')
file.write('Temperature: ' + str(s_temperature) + '\n')
file.close()

exoplanet = pd.read_csv('exoplanet.csv')
pmass = exoplanet.mass.to_numpy() #[Mjup]
semi_major = exoplanet.semi_major_axis.to_numpy() #semi-major axis [AU]
plt.scatter(np.log10(semi_major), np.log10(pmass), s=10, color='#444444', label='confirmed exoplanets')
plt.scatter(np.log10(orbital_separation/u.au), np.log10(p_mass.to(u.Mjup)/u.Mjup), s=10, color='#FF4242', label='pableton 69 b')
plt.xlabel('Semi-Major Axis [AU]')
plt.ylabel('Planetary Mass [Mjup]')
plt.legend(loc='best')
plt.savefig('exoplanets.png', dpi=300)
plt.show()

'''
smass = exoplanet.star_mass.to_numpy()
plt.scatter(np.log10(smass), np.log10(pmass), s=10, color='#444444', label='confirmed exoplanets')
plt.scatter(np.log10(s_mass/u.Msun), np.log10(p_mass.to(u.Mjup)/u.Mjup), s=10, color='#FF4242', label='pableton 69 b')
plt.xlabel('Planetary Mass*sin(i) [Mjup]')
plt.ylabel('Star Mass [Msun]')
plt.legend(loc='best')
plt.savefig('star_vs_planet.png', dpi=300)
plt.show()
'''