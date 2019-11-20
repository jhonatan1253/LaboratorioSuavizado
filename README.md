# librerias
import Datoscardiacos as sig
import numpy as np
import pylab as plt
from scipy import signal
from scipy.ndimage import gaussian_filter


##### Senales del corazon ###
#####   Visualizar la señal del corazon
sig_input_sensor=sig.sensor
n_1=len(sig_input_sensor)
plt.plot(sig_input_sensor, color='orange', label="señal original del corazon")
plt.plot(sig_input_sensor, color='orange', label="señal original del corazon")
plt.show()

### Mean Average para la señal de corazon
#definir ventana
k=14
m=0
mean_smooth=np.zeros(n_1)#Llenar un vector de tamano n con zeros
for i in range(k+1,n_1-k-1):
    m+=1 
    mean_smooth[i]=np.mean(sig_input_sensor[i-k:m+k])
plt.plot(mean_smooth,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

#### Variacion Mean average 
k=9
mean_smooth1=np.zeros(n_1)
for i in range(int(np.floor(k/2)),int(n_1-np.floor(k/2)-1)):
    mean_smooth1[i]=491
    for j in range(int(-np.floor(k/2)),int(np.floor(k/2))):
        mean_smooth1[i]=mean_smooth1[i]+sig_input_sensor[i+j]
    mean_smooth1[i]=mean_smooth1[i]/k
    
plt.plot(mean_smooth1,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

###Median
k=11
m=0
median_smooth=np.zeros(n_1)
for i in range (k+1,n_1-k-1):
    m+=1
    median_smooth[i]=np.median(sig_input_sensor[i-k:m+k])
    
plt.plot(median_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()

#### Savitzky Golay

savis_smooth=signal.savgol_filter(sig_input_sensor,9,3) ##Señal de entrada, tamaño, exponente
signal.savgol_coeffs(11,3)
plt.plot(savis_smooth,color='black', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()


####### Gaussian smmothing

gaus_smooth=gaussian_filter(sig_input_sensor,2)
plt.plot(gaus_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()


# librerias
import DatosFlex as sig
import numpy as np
import pylab as plt
from scipy import signal
from scipy.ndimage import gaussian_filter


##### Senales sensor flexible
#####   Visualizar la señal del  sensor fexible
sig_input_sensor=sig.sensor
n_1=len(sig_input_sensor)
plt.plot(sig_input_sensor, color='orange', label="señal original del corazon")
plt.plot(sig_input_sensor, color='orange', label="señal original del corazon")
plt.show()

### Mean Average para la señal de corazon
#definir ventana
k=14
m=0
mean_smooth=np.zeros(n_1)#Llenar un vector de tamano n con zeros
for i in range(k+1,n_1-k-1):
    m+=1 
    mean_smooth[i]=np.mean(sig_input_sensor[i-k:m+k])
plt.plot(mean_smooth,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

#### Variacion Mean average 
k=9
mean_smooth1=np.zeros(n_1)
for i in range(int(np.floor(k/2)),int(n_1-np.floor(k/2)-1)):
    mean_smooth1[i]=491
    for j in range(int(-np.floor(k/2)),int(np.floor(k/2))):
        mean_smooth1[i]=mean_smooth1[i]+sig_input_sensor[i+j]
    mean_smooth1[i]=mean_smooth1[i]/k
    
plt.plot(mean_smooth1,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

###Median
k=11
m=0
median_smooth=np.zeros(n_1)
for i in range (k+1,n_1-k-1):
    m+=1
    median_smooth[i]=np.median(sig_input_sensor[i-k:m+k])
    
plt.plot(median_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()

#### Savitzky Golay

savis_smooth=signal.savgol_filter(sig_input_sensor,9,3) ##Señal de entrada, tamaño, exponente
signal.savgol_coeffs(11,3)
plt.plot(savis_smooth,color='black', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()


####### Gaussian smmothing

gaus_smooth=gaussian_filter(sig_input_sensor,2)
plt.plot(gaus_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()

# librerias
import DatosFuer as sig
import numpy as np
import pylab as plt
from scipy import signal
from scipy.ndimage import gaussian_filter


##### Señales  ###
#####   Visualizar la señal 
sig_input_sensor=sig.sensor
n_1=len(sig_input_sensor)
plt.plot(sig_input_sensor, color='orange' )
plt.plot(sig_input_sensor, color='orange' )
plt.show()

### Mean Average para la señal 
#definir ventana
k=14
m=0
mean_smooth=np.zeros(n_1)#Llenar un vector de tamano n con zeros
for i in range(k+1,n_1-k-1):
    m+=1 
    mean_smooth[i]=np.mean(sig_input_sensor[i-k:m+k])
plt.plot(mean_smooth,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

#### Variacion Mean average 
k=9
mean_smooth1=np.zeros(n_1)
for i in range(int(np.floor(k/2)),int(n_1-np.floor(k/2)-1)):
    mean_smooth1[i]=491
    for j in range(int(-np.floor(k/2)),int(np.floor(k/2))):
        mean_smooth1[i]=mean_smooth1[i]+sig_input_sensor[i+j]
    mean_smooth1[i]=mean_smooth1[i]/k
    
plt.plot(mean_smooth1,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

###Median
k=11
m=0
median_smooth=np.zeros(n_1)
for i in range (k+1,n_1-k-1):
    m+=1
    median_smooth[i]=np.median(sig_input_sensor[i-k:m+k])
    
plt.plot(median_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()

#### Savitzky Golay

savis_smooth=signal.savgol_filter(sig_input_sensor,9,3) ##Señal de entrada, tamaño, exponente
signal.savgol_coeffs(11,3)
plt.plot(savis_smooth,color='black', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()


####### Gaussian smmothing

gaus_smooth=gaussian_filter(sig_input_sensor,2)
plt.plot(gaus_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()

import DatosPresion as sig
import numpy as np
import pylab as plt
from scipy import signal
from scipy.ndimage import gaussian_filter


##### Senales de fuerza 
#####   Visualizar la señal de fuerza
sig_input_sensor=sig.sensor
n_1=len(sig_input_sensor)
plt.plot(sig_input_sensor, color='orange', label="señal original del corazon")
plt.plot(sig_input_sensor, color='orange', label="señal original del corazon")
plt.show()

### Mean Average para la señal   fuerza
#definir ventana
k=14
m=0
mean_smooth=np.zeros(n_1)#Llenar un vector de tamano n con zeros
for i in range(k+1,n_1-k-1):
    m+=1 
    mean_smooth[i]=np.mean(sig_input_sensor[i-k:m+k])
plt.plot(mean_smooth,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

#### Variacion Mean average 
k=9
mean_smooth1=np.zeros(n_1)
for i in range(int(np.floor(k/2)),int(n_1-np.floor(k/2)-1)):
    mean_smooth1[i]=491
    for j in range(int(-np.floor(k/2)),int(np.floor(k/2))):
        mean_smooth1[i]=mean_smooth1[i]+sig_input_sensor[i+j]
    mean_smooth1[i]=mean_smooth1[i]/k
    
plt.plot(mean_smooth1,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

###Median
k=11
m=0
median_smooth=np.zeros(n_1)
for i in range (k+1,n_1-k-1):
    m+=1
    median_smooth[i]=np.median(sig_input_sensor[i-k:m+k])
    
plt.plot(median_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()

#### Savitzky Golay

savis_smooth=signal.savgol_filter(sig_input_sensor,9,3) ##Señal de entrada, tamaño, exponente
signal.savgol_coeffs(11,3)
plt.plot(savis_smooth,color='black', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()


####### Gaussian smmothing

gaus_smooth=gaussian_filter(sig_input_sensor,2)
plt.plot(gaus_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()

# librerias
import DatosPresion as sig
import numpy as np
import pylab as plt
from scipy import signal
from scipy.ndimage import gaussian_filter


##### Senales de presion ###
#####   Visualizar la señal de presion 
sig_input_sensor=sig.sensor
n_1=len(sig_input_sensor)
plt.plot(sig_input_sensor, color='orange', label="señal original del corazon")
plt.plot(sig_input_sensor, color='orange', label="señal original del corazon")
plt.show()

### Mean Average para la señal presion
#definir ventana
k=14
m=0
mean_smooth=np.zeros(n_1)#Llenar un vector de tamano n con zeros
for i in range(k+1,n_1-k-1):
    m+=1 
    mean_smooth[i]=np.mean(sig_input_sensor[i-k:m+k])
plt.plot(mean_smooth,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

#### Variacion Mean average 
k=9
mean_smooth1=np.zeros(n_1)
for i in range(int(np.floor(k/2)),int(n_1-np.floor(k/2)-1)):
    mean_smooth1[i]=491
    for j in range(int(-np.floor(k/2)),int(np.floor(k/2))):
        mean_smooth1[i]=mean_smooth1[i]+sig_input_sensor[i+j]
    mean_smooth1[i]=mean_smooth1[i]/k
    
plt.plot(mean_smooth1,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

###Median
k=11
m=0
median_smooth=np.zeros(n_1)
for i in range (k+1,n_1-k-1):
    m+=1
    median_smooth[i]=np.median(sig_input_sensor[i-k:m+k])
    
plt.plot(median_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()

#### Savitzky Golay

savis_smooth=signal.savgol_filter(sig_input_sensor,9,3) ##Señal de entrada, tamaño, exponente
signal.savgol_coeffs(11,3)
plt.plot(savis_smooth,color='black', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()


####### Gaussian smmothing

gaus_smooth=gaussian_filter(sig_input_sensor,2)
plt.plot(gaus_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()


# librerias
import DatosRespir as sig
import numpy as np
import pylab as plt
from scipy import signal
from scipy.ndimage import gaussian_filter


##### Senales de respiracion ###
#####   Visualizar la señal de respiracion
sig_input_sensor=sig.sensor
n_1=len(sig_input_sensor)
plt.plot(sig_input_sensor, color='orange', label="señal original del corazon")
plt.plot(sig_input_sensor, color='orange', label="señal original del corazon")
plt.show()

### Mean Average para la señal de respiracion
#definir ventana
k=14
m=0
mean_smooth=np.zeros(n_1)#Llenar un vector de tamano n con zeros
for i in range(k+1,n_1-k-1):
    m+=1 
    mean_smooth[i]=np.mean(sig_input_sensor[i-k:m+k])
plt.plot(mean_smooth,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

#### Variacion Mean average 
k=9
mean_smooth1=np.zeros(n_1)
for i in range(int(np.floor(k/2)),int(n_1-np.floor(k/2)-1)):
    mean_smooth1[i]=491
    for j in range(int(-np.floor(k/2)),int(np.floor(k/2))):
        mean_smooth1[i]=mean_smooth1[i]+sig_input_sensor[i+j]
    mean_smooth1[i]=mean_smooth1[i]/k
    
plt.plot(mean_smooth1,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

###Median
k=11
m=0
median_smooth=np.zeros(n_1)
for i in range (k+1,n_1-k-1):
    m+=1
    median_smooth[i]=np.median(sig_input_sensor[i-k:m+k])
    
plt.plot(median_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()

#### Savitzky Golay

savis_smooth=signal.savgol_filter(sig_input_sensor,9,3) ##Señal de entrada, tamaño, exponente
signal.savgol_coeffs(11,3)
plt.plot(savis_smooth,color='black', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()


####### Gaussian smmothing

gaus_smooth=gaussian_filter(sig_input_sensor,2)
plt.plot(gaus_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()



# librerias
import DatosUltra as sig
import numpy as np
import pylab as plt
from scipy import signal
from scipy.ndimage import gaussian_filter


##### Senales del ultrasonido ###
#####   Visualizar la señal de ultrasonido
sig_input_sensor=sig.sensor
n_1=len(sig_input_sensor)
plt.plot(sig_input_sensor, color='orange', label="señal original del corazon")
plt.plot(sig_input_sensor, color='orange', label="señal original del corazon")
plt.show()

### Mean Average para la señal de ultrasonido
#definir ventana
k=14
m=0
mean_smooth=np.zeros(n_1)#Llenar un vector de tamano n con zeros
for i in range(k+1,n_1-k-1):
    m+=1 
    mean_smooth[i]=np.mean(sig_input_sensor[i-k:m+k])
plt.plot(mean_smooth,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

#### Variacion Mean average 
k=9
mean_smooth1=np.zeros(n_1)
for i in range(int(np.floor(k/2)),int(n_1-np.floor(k/2)-1)):
    mean_smooth1[i]=491
    for j in range(int(-np.floor(k/2)),int(np.floor(k/2))):
        mean_smooth1[i]=mean_smooth1[i]+sig_input_sensor[i+j]
    mean_smooth1[i]=mean_smooth1[i]/k
    
plt.plot(mean_smooth1,color='blue', label ='data smoothing')
plt.plot(sig_input_sensor,color='orange',label='original samples')
plt.show()

###Median
k=11
m=0
median_smooth=np.zeros(n_1)
for i in range (k+1,n_1-k-1):
    m+=1
    median_smooth[i]=np.median(sig_input_sensor[i-k:m+k])
    
plt.plot(median_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()

#### Savitzky Golay

savis_smooth=signal.savgol_filter(sig_input_sensor,9,3) ##Señal de entrada, tamaño, exponente
signal.savgol_coeffs(11,3)
plt.plot(savis_smooth,color='black', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()


####### Gaussian smmothing

gaus_smooth=gaussian_filter(sig_input_sensor,2)
plt.plot(gaus_smooth,color='blue', label="data smoothing") 
plt.plot(sig_input_sensor, color='orange', label="original sample")
plt.show()
